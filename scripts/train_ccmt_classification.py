from pathlib import Path
import sys
from typing import Optional
import json
from datetime import datetime
import time
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
)

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from src.models import load_ccmt_only_model
from src.data.dataset import MSP_Podcast_Dataset
from src.data.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset
from src.utils.config import get_training_config
from transformers import get_linear_schedule_with_warmup


SUPPORTED_MODALITIES = ["text_en", "text_es", "text_de", "text_fr", "audio"]


def parse_modalities(modalities_arg: Optional[str]) -> list[str]:
    if not modalities_arg:
        return ["text_en", "text_es", "audio"]

    modalities = [item.strip() for item in modalities_arg.split(",") if item.strip()]
    invalid_modalities = [item for item in modalities if item not in SUPPORTED_MODALITIES]
    if invalid_modalities:
        raise ValueError(
            f"Modalitati invalide: {invalid_modalities}. Alege dintre {SUPPORTED_MODALITIES}"
        )
    if "audio" not in modalities:
        raise ValueError("Configuratiile CCMT suportate aici trebuie sa includa modalitatea 'audio'")
    if "text_en" not in modalities:
        raise ValueError("Configuratiile CCMT cerute trebuie sa includa modalitatea 'text_en'")
    return modalities


def build_modality_suffix(modalities: list[str]) -> str:
    return "_".join(modalities)


def resolve_embeddings_dir(embeddings_dir_arg: Optional[str], modalities: list[str]) -> Path:
    if embeddings_dir_arg:
        return Path(embeddings_dir_arg)

    if modalities == ["text_en", "text_es", "audio"]:
        return PROJECT_ROOT / "MSP_Podcast" / "embeddings"

    return PROJECT_ROOT / "MSP_Podcast" / f"embeddings_{build_modality_suffix(modalities)}"


def collect_embedding_inputs(
    batch: dict,
    modalities: list[str],
    device: str,
    amp_dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    inputs = {}
    for modality in modalities:
        tensor = batch[f"{modality}_emb"]
        if amp_dtype is not None:
            inputs[f"{modality}_emb"] = tensor.to(device, dtype=amp_dtype, non_blocking=True)
        else:
            inputs[f"{modality}_emb"] = tensor.to(device, non_blocking=True)
    return inputs


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
        checkpoint_dir: Path = Path("checkpoints/ccmt_multimodal"),
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
            'loss_function': 'CrossEntropyLoss',
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
        
        # Loss function: CrossEntropyLoss cu ponderi

        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_f1 = 0.0
        self.best_epoch: Optional[int] = None
        self.patience_counter = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
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
        """Train pentru o epocă."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        self.optimizer.zero_grad(set_to_none=True)
        
        # Setăm formatul optim pentru AMP (BFloat16 pt seria RTX 40xx, Float16 altfel)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        for batch_idx, batch in enumerate(progress_bar):
            embedding_inputs = collect_embedding_inputs(
                batch=batch,
                modalities=self.model_config['modalities'],
                device=self.device,
                amp_dtype=amp_dtype,
            )
            
            # Etichetele rămân întregi (long) pentru CrossEntropyLoss!
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Mixed precision context modern
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.use_amp):
                # Forward pass
                predictions = self.model(**embedding_inputs)
                
                # CrossEntropyLoss știe să gestioneze logits în FP16/BF16, dar e mai sigur cu float32 intern
                loss = self.criterion(predictions.float(), labels)
                
                # Scalează loss-ul pentru gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass (acumulează gradienți)
            if self.use_amp:
                self.scaler.scale(loss).backward()  # type: ignore
            else:
                loss.backward()
            
            # Update parametrii doar la fiecare N steps
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
            
            # Metrics (folosește loss-ul nescalat pentru logging)
            total_loss += loss.item() * self.gradient_accumulation_steps
            pred_classes = predictions.argmax(dim=1).cpu().numpy()
            true_classes = labels.cpu().numpy()
            
            all_preds.extend(pred_classes)
            all_labels.extend(true_classes)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Final step dacă ultimul batch nu a declanșat update-ul
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
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_macro': epoch_f1,
        }

    @torch.no_grad()
    def evaluate(self, loader, desc="Validation") -> dict:
        """Evaluare pe un dataset."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(loader, desc=desc)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        for batch in progress_bar:
            embedding_inputs = collect_embedding_inputs(
                batch=batch,
                modalities=self.model_config['modalities'],
                device=self.device,
                amp_dtype=amp_dtype,
            )
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Mixed precision pentru inferență
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.use_amp):
                predictions = self.model(**embedding_inputs)
                
                loss = self.criterion(predictions.float(), labels)
            
            total_loss += loss.item()
            
            pred_classes = predictions.argmax(dim=1).cpu().numpy()
            true_classes = labels.cpu().numpy()
            
            all_preds.extend(pred_classes)
            all_labels.extend(true_classes)
        
        epoch_loss = total_loss / len(loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_macro': epoch_f1,
            'predictions': all_preds,
            'labels': all_labels,
        }

    def train(self):
        """Training loop complet."""
        train_start_time = time.time()

        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader, desc="Validation")
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1_macro'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1_macro']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1_macro']:.4f}")
            
            #self.scheduler.step(val_metrics['f1_macro'])
            
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ New best model! F1: {self.best_val_f1:.4f}")
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
        print(f"  Accuracy: {test_metrics_full['accuracy']:.4f}")
        print(f"  F1 Macro: {test_metrics_full['f1_macro']:.4f}")
        print(f"  F1 Weighted: {test_metrics_full['f1_weighted']:.4f}")
        print(f"  Precision Macro: {test_metrics_full['precision_macro']:.4f}")
        print(f"  Recall Macro: {test_metrics_full['recall_macro']:.4f}")
        print(f"  Balanced Accuracy: {test_metrics_full['balanced_accuracy']:.4f}")
        print(f"  Loss: {test_metrics_full['loss']:.4f}")
        print(f"  Training Time: {train_time_minutes:.2f} minutes")
        
        print("\nClassification Report:")
        print(classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=['unsatisfied', 'neutral', 'satisfied']
        ))
        
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
            'best_val_f1': self.best_val_f1,
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
        """Calculează set complet de metrici pe test pentru best model."""
        y_true = test_metrics['labels']
        y_pred = test_metrics['predictions']

        return {
            'loss': test_metrics['loss'],
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'classification_report': classification_report(
                y_true,
                y_pred,
                target_names=['unsatisfied', 'neutral', 'satisfied'],
                output_dict=True,
                zero_division=0,
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'num_samples': len(y_true),
        }

    def save_test_metrics(self, test_metrics: dict, train_time_minutes: float):
        """Salvează metricile complete pe test pentru best model."""
        payload = {
            'timestamp': datetime.now().isoformat(),
            'best_epoch': self.best_epoch,
            'best_val_f1': self.best_val_f1,
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
    parser = argparse.ArgumentParser(description="Train CCMT classification with selectable modalities")
    parser.add_argument(
        "--modalities",
        type=str,
        default="text_en,text_es,audio",
        help="Lista de modalitati separate prin virgula. Exemple: text_en,audio sau text_en,text_fr,audio",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Director pentru checkpoint-uri. Daca lipseste, se genereaza automat din modalitati.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Directorul cu embeddings precompute. Implicit: MSP_Podcast/embeddings pentru text_en,text_es,audio, altfel MSP_Podcast/embeddings_<modalitati>",
    )
    args = parser.parse_args()

    train_config = get_training_config(
        "ccmt_classification",
        PROJECT_ROOT / "configs" / "training_config.json",
    )
    modalities = parse_modalities(args.modalities)
    modality_suffix = build_modality_suffix(modalities)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = int(train_config["batch_size"])
    GRADIENT_ACCUMULATION_STEPS = int(train_config["gradient_accumulation_steps"])
    LEARNING_RATE = float(train_config["learning_rate"])
    NUM_EPOCHS = int(train_config["num_epochs"])
    USE_AMP = bool(train_config["use_amp"])
    USE_COMPILE = bool(train_config["use_compile"])
    NUM_WORKERS = int(train_config["num_workers"])
    
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
    embeddings_dir = resolve_embeddings_dir(args.embeddings_dir, modalities)
    print(f"Embeddings dir: {embeddings_dir}")
    sample_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='train',
        device='cpu',
        modalities=modalities,
    )
    embedding_dims = sample_dataset.get_embedding_dims()

    model_config = {
        'num_classes': 3,
        'text_en_dim': embedding_dims.get('text_en', 768),
        'text_es_dim': embedding_dims.get('text_es', 768),
        'text_de_dim': embedding_dims.get('text_de', 768),
        'text_fr_dim': embedding_dims.get('text_fr', 768),
        'audio_dim': embedding_dims.get('audio', 768),
        'ccmt_dim': 768,
        'num_patches_per_modality': 100,
        'ccmt_depth': 4,
        'ccmt_heads': 4,
        'ccmt_mlp_dim': 1024,
        'ccmt_dropout': 0.1,
        'modalities': modalities,
    }
    
    print("Loading CCMT model (no backbones - using precomputed embeddings)...")
    model = load_ccmt_only_model(
        text_en_dim=model_config['text_en_dim'],
        text_es_dim=model_config['text_es_dim'],
        text_de_dim=model_config['text_de_dim'],
        text_fr_dim=model_config['text_fr_dim'],
        audio_dim=model_config['audio_dim'],
        num_classes=model_config['num_classes'],
        ccmt_dim=model_config['ccmt_dim'],
        num_patches_per_modality=model_config['num_patches_per_modality'],
        ccmt_depth=model_config['ccmt_depth'],
        ccmt_heads=model_config['ccmt_heads'],
        ccmt_mlp_dim=model_config['ccmt_mlp_dim'],
        ccmt_dropout=model_config['ccmt_dropout'],
        device=DEVICE,
        modalities=modalities,
    )
    
    # Compilare model (PyTorch 2.0+) - experimental, lăsat False din setări
    if USE_COMPILE and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='default')
        print("✓ Model compiled")
    
    print("\nLoading datasets...")

    # Calculăm class weights din etichetele setului de antrenare MSP-Podcast
    data_dir = PROJECT_ROOT / "MSP_Podcast"
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    class_weights_dataset = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition='Train',
        modalities=[], #ca sa nu incarcam datele reale
    )
    all_train_labels = class_weights_dataset.metadata['label_id'].astype(int).to_numpy()
    raw_weights = class_weights_dataset.get_class_weights(all_train_labels, device=DEVICE)
    # 2. Aplicăm NETEZIRE cu Radical (Square Root)
    class_weights = torch.sqrt(raw_weights)
    print(f"Smoothed Class weights (Train): {class_weights.detach().cpu().tolist()}")

    #class_weights = None
    
    # Încarcă embeddingurile precalculate
    train_dataset = sample_dataset
    
    val_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='val',
        device='cpu',
        modalities=modalities,
    )
    
    # Pentru test, verificăm dacă există test1 (conform partițiilor MSP-Podcast)
    test_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='val',  # Folosim val ca fallback dacă nu există test separat
        device='cpu',
        modalities=modalities,
    )
    
    # Creăm DataLoaders (Am lăsat num_workers=0, standard pentru stabilitate pe Windows la citire din memorie)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    print(f"\n✓ Datasets loaded:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints") / f"ccmt_multimodal_{modality_suffix}"

    trainer = CCMTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=5,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        use_amp=USE_AMP,
        class_weights=class_weights,
        batch_size=BATCH_SIZE,
        model_config=model_config,
    )
    
    test_metrics = trainer.train()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final Test F1: {test_metrics['f1_macro']:.4f}")
    print(f"Final Test Acc: {test_metrics['accuracy']:.4f}")

if __name__ == '__main__':
    main()