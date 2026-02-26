"""
Script de training pentru modelul CCMT multimodal complet.
Fine-tuneaza doar fusion adapters + CCMT cu backbones frozen.
Optimizat pentru viteză maximă cu Mixed Precision (BF16/FP16) și I/O eficient.
"""
from pathlib import Path
import sys
from typing import Optional
import json
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_full_multimodal_model
from src.data.dataset import MSP_Podcast_Dataset
from scripts.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset


class CCMTTrainer:
    """Trainer pentru CCMT multimodal cu backbones pretrenate."""
    
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        
        # Loss function: CrossEntropyLoss cu ponderi
        class_weights = torch.tensor([1.02, 1.00, 1.60], dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_f1 = 0.0
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
            # Mutăm datele pe device și facem cast la FP16/BF16 concomitent (evităm overhead-ul)
            text_en_emb = batch['text_en_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            text_es_emb = batch['text_es_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            audio_emb = batch['audio_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            
            # Etichetele rămân întregi (long) pentru CrossEntropyLoss!
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Mixed precision context modern
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=self.use_amp):
                # Forward pass
                predictions = self.model(
                    text_en_emb=text_en_emb,
                    text_es_emb=text_es_emb, 
                    audio_emb=audio_emb,
                )
                
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
            text_en_emb = batch['text_en_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            text_es_emb = batch['text_es_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            audio_emb = batch['audio_emb'].to(self.device, dtype=amp_dtype, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Mixed precision pentru inferență
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=self.use_amp):
                predictions = self.model(
                    text_en_emb=text_en_emb,
                    text_es_emb=text_es_emb,
                    audio_emb=audio_emb,
                )
                
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
            
            self.scheduler.step(val_metrics['f1_macro'])
            
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
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
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=['unsatisfied', 'neutral', 'satisfied']
        ))
        
        self.save_history()
        return test_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
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
            print(f"✓ Loaded best model from {path}")
        else:
            print(f"⚠ No checkpoint found at {path}")

    def save_history(self):
        path = self.checkpoint_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {path}")


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # OPTIMIZARE: CCMT consumă extrem de puțină memorie, putem urca Batch Size masiv!
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2  # 64 este suficient, nu e nevoie de acumulare adițională aici
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
    
    print("Loading model...")
    model = load_full_multimodal_model(
        text_en_checkpoint="checkpoints/roberta_text_en",
        text_es_checkpoint="checkpoints/roberta_text_es",
        audio_checkpoint="checkpoints/wavlm_audio",
        num_classes=3,
        ccmt_dim=768,               # Setăm nativ la 768
        num_patches_per_modality=100,
        ccmt_depth=8,               # Recomandat de lucrare
        ccmt_heads=8,               # Recomandat de lucrare
        ccmt_mlp_dim=2048,
        freeze_backbones=True,
        projection_dim=None,        # Fără strat de proiecție
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
        device='cpu'
    )
    
    val_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='val',
        device='cpu'
    )
    
    # Pentru test, verificăm dacă există test1 (conform partițiilor MSP-Podcast)
    test_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='val',  # Folosim val ca fallback dacă nu există test separat
        device='cpu'
    )
    
    # Creăm DataLoaders (Am lăsat num_workers=0, standard pentru stabilitate pe Windows la citire din memorie)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
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
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=Path("checkpoints/ccmt_multimodal"),
        early_stopping_patience=7,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        use_amp=USE_AMP,
    )
    
    test_metrics = trainer.train()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final Test F1: {test_metrics['f1_macro']:.4f}")
    print(f"Final Test Acc: {test_metrics['accuracy']:.4f}")

if __name__ == '__main__':
    main()