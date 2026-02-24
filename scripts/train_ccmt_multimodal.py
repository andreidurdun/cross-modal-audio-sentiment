"""
Script de training pentru modelul CCMT multimodal complet.
Fine-tuneaza doar fusion adapters + CCMT cu backbones frozen.
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
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
            verbose=True
        )
        
        # Loss function corectat: CrossEntropyLoss cu ponderi
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
        # Presupunem ca ai o functie print_parameter_summary, daca nu, o poti comenta
        if hasattr(self.model, 'print_parameter_summary'):
            self.model.print_parameter_summary()
        print(f"Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
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
        
        for batch_idx, batch in enumerate(progress_bar):
            # Tensi de forma [batch_size, 100, 768]
            text_en_emb = batch['text_en_emb'].to(self.device)
            text_es_emb = batch['text_es_emb'].to(self.device)
            audio_emb = batch['audio_emb'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass (Așteptăm logits necruntați prin Sigmoid/Softmax)
            predictions = self.model(
                text_en_emb=text_en_emb,
                text_es_emb=text_es_emb, 
                audio_emb=audio_emb,
            )
            
            # Calculăm loss-ul (CrossEntropyLoss acceptă direct indicii claselor)
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred_classes = predictions.argmax(dim=1).cpu().numpy()
            true_classes = labels.cpu().numpy()
            
            all_preds.extend(pred_classes)
            all_labels.extend(true_classes)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
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
        
        for batch in progress_bar:
            text_en_emb = batch['text_en_emb'].to(self.device)
            text_es_emb = batch['text_es_emb'].to(self.device)
            audio_emb = batch['audio_emb'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            predictions = self.model(
                text_en_emb=text_en_emb,
                text_es_emb=text_es_emb,
                audio_emb=audio_emb,
            )
            
            loss = self.criterion(predictions, labels)
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
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    
    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    print("\nLoading model...")
    model = load_full_multimodal_model(
        text_en_checkpoint="checkpoints/roberta_text_en",
        text_es_checkpoint="checkpoints/roberta_text_es",
        audio_checkpoint="checkpoints/wavlm_audio",
        num_classes=3,
        ccmt_dim=768,               # CORECTAT: Setăm nativ la 768
        num_patches_per_modality=100,
        ccmt_depth=8,               # Recomandat de lucrare
        ccmt_heads=8,               # Recomandat de lucrare
        ccmt_mlp_dim=2048,
        freeze_backbones=True,
        projection_dim=None,        # CORECTAT: Fără strat de proiecție
        device=DEVICE,
    )
    
    print("\n⚠ Demo mode: Creating dummy dataloaders")
    from torch.utils.data import TensorDataset
    
    n_train, n_val, n_test = 1000, 200, 200
    
    def create_dummy_loader(n_samples, batch_size):
        # CORECTAT: Tensorii trebuie să fie 3D (batch_size, num_patches, embed_dim)
        text_en_emb = torch.randn(n_samples, 100, 768)
        text_es_emb = torch.randn(n_samples, 100, 768)
        audio_emb = torch.randn(n_samples, 100, 768)
        labels = torch.randint(0, 3, (n_samples,))
        
        dataset = TensorDataset(text_en_emb, text_es_emb, audio_emb, labels)
        
        class BatchWrapper:
            def __init__(self, loader):
                self.loader = loader
            def __len__(self):
                return len(self.loader)
            def __iter__(self):
                for text_en, text_es, audio, labels in self.loader:
                    yield {
                        'text_en_emb': text_en,
                        'text_es_emb': text_es,
                        'audio_emb': audio,
                        'labels': labels,
                    }
                    
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return BatchWrapper(loader)
    
    train_loader = create_dummy_loader(n_train, BATCH_SIZE)
    val_loader = create_dummy_loader(n_val, BATCH_SIZE)
    test_loader = create_dummy_loader(n_test, BATCH_SIZE)
    
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
    )
    
    test_metrics = trainer.train()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final Test F1: {test_metrics['f1_macro']:.4f}")
    print(f"Final Test Acc: {test_metrics['accuracy']:.4f}")

if __name__ == '__main__':
    main()