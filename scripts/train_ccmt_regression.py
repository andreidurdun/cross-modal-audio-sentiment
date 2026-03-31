from pathlib import Path
import sys
from typing import Optional
import json
from datetime import datetime
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_ccmt_only_model
from scripts.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset
from transformers import get_linear_schedule_with_warmup


def concordance_correlation_coefficient(y_true, y_pred):
    """Returnează CCC pentru fiecare coloană și media lor (pentru shape [N,2])."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true, axis=0)
    mean_pred = np.mean(y_pred, axis=0)
    var_true = np.var(y_true, axis=0)
    var_pred = np.var(y_pred, axis=0)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred), axis=0)
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    if hasattr(ccc, 'shape') and ccc.shape:
        return {
            'valence': float(ccc[0]),
            'arousal': float(ccc[1]),
            'mean': float(np.mean(ccc))
        }
    return {'valence': float(ccc), 'arousal': float(ccc), 'mean': float(ccc)}


class CCMTTrainer:
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 50,
        checkpoint_dir: Path = Path("checkpoints/ccmt_multimodal_regresion"),
        early_stopping_patience: int = 7,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        model_config: Optional[dict] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
        
        # Store hyperparameters pentru salvare
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'use_amp': use_amp,
            'batch_size': batch_size,
            'effective_batch_size': batch_size * gradient_accumulation_steps,
            'optimizer': 'AdamW',
            'scheduler': 'LinearWithWarmup',
            'loss_function': 'MSELoss',
        }
        
        # Store model configuration
        self.model_config = model_config or {}
        
        # Mixed Precision Training
        self.use_amp = use_amp and (device == "cuda")
        self.scaler: Optional[torch.cuda.amp.GradScaler] = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Optimizer - doar parametrii trainable
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * 0.1)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps, # 10% din antrenament e încălzire
            num_training_steps=total_steps
        )
        

        # Loss function: MSE pentru regresie (valence, arousal)
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.best_epoch: Optional[int] = None
        self.patience_counter = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        print(f"\n{'='*70}")
        print("CCMT TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
        print(f"Mixed Precision (AMP): {self.use_amp}")
        print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        effective_batch_size = train_loader.batch_size * gradient_accumulation_steps
        print(f"Effective Batch Size: {effective_batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"{'='*70}\n")

    def train_epoch(self) -> dict:
        """Train pentru o epocă (regresie valence/arousal)."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        progress_bar = tqdm(self.train_loader, desc="Training")
        self.optimizer.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        for batch_idx, batch in enumerate(progress_bar):
            text_en_emb = batch['text_en_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            text_es_emb = batch['text_es_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            audio_emb = batch['audio_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            # labels: shape [batch, 2] (valence, arousal)
            labels = batch['val_arousal'].to(self.device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.use_amp):
                predictions = self.model(
                    text_en_emb=text_en_emb,
                    text_es_emb=text_es_emb, 
                    audio_emb=audio_emb,
                )
                loss = self.criterion(predictions.float(), labels)
                loss = loss / self.gradient_accumulation_steps
            if self.use_amp:
                self.scaler.scale(loss).backward()  # type: ignore
            else:
                loss.backward()
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)  # type: ignore
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)  # type: ignore
                    self.scaler.update()  # type: ignore
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = predictions.detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labs)
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)  # type: ignore
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)  # type: ignore
                self.scaler.update()  # type: ignore
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        epoch_loss = total_loss / len(self.train_loader)
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        ccc = concordance_correlation_coefficient(all_labels, all_preds)
        return {
            'loss': epoch_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'ccc_valence': ccc['valence'],
            'ccc_arousal': ccc['arousal'],
            'ccc_mean': ccc['mean'],
        }

    @torch.no_grad()
    def evaluate(self, loader, desc="Validation") -> dict:
        """Evaluare pe un dataset (regresie valence/arousal)."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        progress_bar = tqdm(loader, desc=desc)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        for batch in progress_bar:
            text_en_emb = batch['text_en_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            text_es_emb = batch['text_es_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            audio_emb = batch['audio_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            labels = batch['val_arousal'].to(self.device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.use_amp):
                predictions = self.model(
                    text_en_emb=text_en_emb,
                    text_es_emb=text_es_emb,
                    audio_emb=audio_emb,
                )
                loss = self.criterion(predictions.float(), labels)
            total_loss += loss.item()
            preds = predictions.detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labs)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        epoch_loss = total_loss / len(loader)
        # Denormalize predictions and labels for MAE/MSE/R2/CCC
        denorm_preds = PrecomputedEmbeddingsDataset.denormalize_val_arousal(torch.from_numpy(all_preds)).numpy()
        denorm_labels = PrecomputedEmbeddingsDataset.denormalize_val_arousal(torch.from_numpy(all_labels)).numpy()
        mse = mean_squared_error(denorm_labels, denorm_preds)
        mae = mean_absolute_error(denorm_labels, denorm_preds)
        r2 = r2_score(denorm_labels, denorm_preds)
        ccc = concordance_correlation_coefficient(denorm_labels, denorm_preds)
        return {
            'loss': epoch_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'ccc_valence': ccc['valence'],
            'ccc_arousal': ccc['arousal'],
            'ccc_mean': ccc['mean'],
            'predictions': denorm_preds,
            'labels': denorm_labels,
        }

    def train(self):
        """Training loop complet pentru regresie valence/arousal."""
        train_start_time = time.time()
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}\n")
        # self.best_val_loss is now used for tracking best validation loss
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader, desc="Validation")
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}, CCC(val): {train_metrics['ccc_valence']:.4f}, CCC(ar): {train_metrics['ccc_arousal']:.4f}, CCC(mean): {train_metrics['ccc_mean']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, R2: {val_metrics['r2']:.4f}, CCC(val): {val_metrics['ccc_valence']:.4f}, CCC(ar): {val_metrics['ccc_arousal']:.4f}, CCC(mean): {val_metrics['ccc_mean']:.4f}")
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ New best model! Val Loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        print(f"\n{'='*70}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*70}")
        self.load_best_checkpoint()
        test_metrics = self.evaluate(self.test_loader, desc="Testing")
        train_time_minutes = (time.time() - train_start_time) / 60.0
        test_metrics_full = self.compute_test_metrics(test_metrics)
        self.save_test_metrics(test_metrics_full, train_time_minutes)
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics_full['loss']:.4f}")
        print(f"  MSE: {test_metrics_full['mse']:.4f}")
        print(f"  MAE: {test_metrics_full['mae']:.4f}")
        print(f"  R2: {test_metrics_full['r2']:.4f}")
        print(f"  CCC(val): {test_metrics_full['ccc_valence']:.4f}")
        print(f"  CCC(ar): {test_metrics_full['ccc_arousal']:.4f}")
        print(f"  CCC(mean): {test_metrics_full['ccc_mean']:.4f}")
        print(f"  Training Time: {train_time_minutes:.2f} minutes")
        self.save_history()
        self.save_loss_plot()
        self.save_config()
        return test_metrics_full

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)
            print(f"  Saved best model to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, path)

    def load_best_checkpoint(self):
        path = self.checkpoint_dir / 'best_model.pt'
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_epoch = checkpoint.get('best_epoch', self.best_epoch)
            print(f"✓ Loaded best model from {path}")
        else:
            print(f"⚠ No checkpoint found at {path}")

    def compute_test_metrics(self, test_metrics: dict) -> dict:
        """Calculează metrici de regresie pentru best model."""
        y_true = test_metrics['labels']
        y_pred = test_metrics['predictions']
        ccc = concordance_correlation_coefficient(y_true, y_pred)
        return {
            'loss': test_metrics['loss'],
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'ccc_valence': ccc['valence'],
            'ccc_arousal': ccc['arousal'],
            'ccc_mean': ccc['mean'],
            'num_samples': len(y_true),
        }

    def save_test_metrics(self, test_metrics: dict, train_time_minutes: float):
        """Salvează metricile complete pe test pentru best model."""
        payload = {
            'timestamp': datetime.now().isoformat(),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'training_time_minutes': train_time_minutes,
            'metrics': test_metrics,
        }

        path = self.checkpoint_dir / 'best_model_test_metrics.json'
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"✓ Saved full test metrics to {path}")

    def save_history(self):
        path = self.checkpoint_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {path}")

    def save_loss_plot(self):
        """Salveaza graficul train/val loss la finalul antrenarii."""
        if not self.history['train_loss'] or not self.history['val_loss']:
            print("⚠ No loss history available for plotting")
            return

        epochs = list(range(1, len(self.history['train_loss']) + 1))
        plot_path = self.checkpoint_dir / 'loss_curve.pdf'

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, format='pdf')
        plt.close()

        print(f"✓ Saved loss plot to {plot_path}")

    def save_config(self):
        """Salvează hiperparametrii de antrenare și parametrii modelului."""
        config = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'training_hyperparameters': self.hyperparameters,
            'model_architecture': self.model_config,
            'scheduler_config': {
                'type': 'LinearWithWarmup',
                'num_warmup_steps': self.warmup_steps,
                'num_training_steps': self.total_steps,
                'warmup_ratio': 0.1,
            },
        }
        
        config_path = self.checkpoint_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved training config to {config_path}")

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # OPTIMIZARE: CCMT consumă extrem de puțină memorie, putem urca Batch Size masiv!
    BATCH_SIZE = 64
    GRADIENT_ACCUMULATION_STEPS = 2 
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    USE_AMP = True  # Mixed Precision Training
    USE_COMPILE = False  # torch.compile() (PyTorch 2.0+, dezactivat pt compatibilitate Windows)
    
    # Optimizări CUDA
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True  # Auto-tune pentru performanță
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 pentru matmul
        torch.backends.cudnn.allow_tf32 = True  # TensorFloat-32 pentru cuDNN
    
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Mixed Precision (AMP): {USE_AMP}")
    print(f"torch.compile(): {USE_COMPILE}")
    if DEVICE == "cuda":
        print(f"cuDNN benchmark: True")
        print(f"TF32 enabled: True")
    print(f"{'='*70}\n")
    
    # Model configuration parameters
    model_config = {
        'num_outputs': 2,  # valence + arousal
        'text_en_dim': 768,
        'text_es_dim': 768,
        'audio_dim': 768,
        'ccmt_dim': 768,
        'num_patches_per_modality': 100,
        'ccmt_depth': 4,
        'ccmt_heads': 4,
        'ccmt_mlp_dim': 1024,
        'ccmt_dropout': 0.1,
    }
    print("Loading CCMT model (no backbones - using precomputed embeddings)...")
    model = load_ccmt_only_model(
        text_en_dim=model_config['text_en_dim'],
        text_es_dim=model_config['text_es_dim'],
        audio_dim=model_config['audio_dim'],
        num_classes=model_config['num_outputs'],
        ccmt_dim=model_config['ccmt_dim'],
        num_patches_per_modality=model_config['num_patches_per_modality'],
        ccmt_depth=model_config['ccmt_depth'],
        ccmt_heads=model_config['ccmt_heads'],
        ccmt_mlp_dim=model_config['ccmt_mlp_dim'],
        ccmt_dropout=model_config['ccmt_dropout'],
        device=DEVICE,
    )
    
    # Compilare model (PyTorch 2.0+) - experimental, lăsat False din setări
    if USE_COMPILE and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='default')
        print("✓ Model compiled")
    
    print("\nLoading datasets...")


    
    # Încarcă embeddingurile precalculate
    embeddings_dir = PROJECT_ROOT / "MSP_Podcast" / "embeddings"
    train_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='train',
        device='cpu',
        regression=True
    )
    val_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='val',
        device='cpu',
        regression=True
    )
    test_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='val',
        device='cpu',
        regression=True
    )
    
    # Creăm DataLoaders (Am lăsat num_workers=0, standard pentru stabilitate pe Windows la citire din memorie)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    print(f"\n✓ Datasets loaded:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    trainer = CCMTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=Path("checkpoints/ccmt_multimodal_regression"),
        early_stopping_patience=7,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        use_amp=USE_AMP,
        batch_size=BATCH_SIZE,
        model_config=model_config,
    )
    
    test_metrics = trainer.train()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()