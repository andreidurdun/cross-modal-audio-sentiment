"""
Script pentru continuarea antrenării modelului CCMT de la un checkpoint salvat.

Folosire:
    python scripts/continue_training_ccmt.py --checkpoint checkpoints/ccmt_multimodal/checkpoint_epoch_10.pt --epochs 20

Argumentele opționale permit:
    - Specificarea checkpoint-ului de pornire
    - Numărul de epoci suplimentare
    - Ajustarea learning rate-ului
    - Modificarea batch size-ului
"""

from pathlib import Path
import sys
from typing import Optional
import json
from datetime import datetime
import argparse
import time

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_ccmt_only_model
from src.data.dataset import MSP_Podcast_Dataset
from scripts.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset
from scripts.train_ccmt_multimodal import CCMTTrainer
from transformers import get_linear_schedule_with_warmup


class CCMTContinuedTrainer(CCMTTrainer):
    """
    Extindere a CCMTTrainer pentru continuarea antrenării de la un checkpoint.
    """
    
    def __init__(self, checkpoint_path: Path, additional_epochs: int = 10, 
                 new_learning_rate: Optional[float] = None,
                 reset_early_stopping: bool = True,
                 **kwargs):
        """
        Inițializare trainer pentru continuarea antrenării.
        
        Args:
            checkpoint_path: Calea către checkpoint-ul de pornire
            additional_epochs: Câte epoci suplimentare să antreneze
            new_learning_rate: Opțional - modifică learning rate-ul
            reset_early_stopping: Dacă să reseteze contorul pentru early stopping
            **kwargs: Argumentele standard pentru CCMTTrainer
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.additional_epochs = additional_epochs
        self.new_learning_rate = new_learning_rate
        self.reset_early_stopping = reset_early_stopping
        
        # Verificare existență checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint-ul nu există: {self.checkpoint_path}")
        
        # Încarcă configurația originală
        checkpoint_dir = self.checkpoint_path.parent
        config_path = checkpoint_dir / 'training_config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            print(f"\n✓ Încărcat configurație existentă din {config_path}")
            
            # Actualizează kwargs cu configurația salvată (dacă nu e deja specificat)
            # Filtrăm doar parametrii validi pentru CCMTTrainer.__init__()
            valid_params = {
                'learning_rate', 'weight_decay', 'num_epochs', 
                'early_stopping_patience', 'gradient_accumulation_steps',
                'use_amp', 'batch_size'
            }
            hyperparams = saved_config.get('training_hyperparameters', {})
            for key, value in hyperparams.items():
                if key in valid_params and key not in kwargs:
                    kwargs[key] = value
            
            # Salvează configurația modelului pentru reîncărcare
            self.saved_model_config = saved_config.get('model_architecture', {})
        else:
            self.saved_model_config = {}
            print(f"⚠ Nu s-a găsit configurația originală în {config_path}")
        
        # Inițializează trainer-ul părinte
        super().__init__(model_config=self.saved_model_config, **kwargs)
        
        # Încarcă checkpoint-ul
        self.start_epoch = self.load_checkpoint()
        
        # Opțional: modifică learning rate-ul
        if new_learning_rate is not None:
            print(f"  Modificare learning rate: {self.optimizer.param_groups[0]['lr']} -> {new_learning_rate}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_learning_rate
            self.hyperparameters['learning_rate'] = new_learning_rate
        
        # Resetează scheduler pentru epocile rămase
        remaining_steps = len(self.train_loader) * additional_epochs // self.gradient_accumulation_steps
        warmup_steps = int(remaining_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=remaining_steps
        )
        print(f"  Scheduler reinițializat: {remaining_steps} pași, {warmup_steps} warmup")
        
        # Opțional: resetează early stopping
        if reset_early_stopping:
            self.patience_counter = 0
            print(f"  Early stopping counter resetat (patience={self.early_stopping_patience})")
        
        print(f"\n{'='*70}")
        print(f"CONTINUARE ANTRENARE")
        print(f"{'='*70}")
        print(f"Checkpoint: {self.checkpoint_path.name}")
        print(f"Epoca de start: {self.start_epoch}")
        print(f"Epoci suplimentare: {additional_epochs}")
        print(f"Epoca finală: {self.start_epoch + additional_epochs - 1}")
        print(f"Best validation F1 din checkpoint: {self.best_val_f1:.4f}")
        print(f"Learning rate curent: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*70}\n")
    
    def load_checkpoint(self) -> int:
        """
        Încarcă checkpoint-ul și restaurează starea trainer-ului.
        
        Returns:
            Epoca de la care se continuă antrenarea (epoch + 1)
        """
        print(f"\n📂 Încărcare checkpoint din {self.checkpoint_path}...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Încarcă starea modelului
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✓ Model state dictionary încărcat")
        
        # Încarcă starea optimizer-ului
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  ✓ Optimizer state dictionary încărcat")
        
        # Încarcă starea scheduler-ului (va fi suprascris mai târziu)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✓ Scheduler state dictionary încărcat")
        
        # Încarcă metrici și istoric
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', None)
        
        # Încarcă istoricul antrenării
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            print(f"  ✓ Istoric antrenare încărcat ({len(self.history['train_loss'])} epoci)")
        
        # Returnează epoca curentă + 1 (pentru continuare)
        current_epoch = checkpoint.get('epoch', 0)
        print(f"  ✓ Se continuă de la epoca {current_epoch + 1}\n")
        
        return current_epoch + 1
    
    def train(self) -> dict:
        """
        Continuă antrenarea pentru epocile suplimentare specificate.
        """
        print(f"🚀 Start antrenare continuată la {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = time.time()
        
        final_epoch = self.start_epoch + self.additional_epochs - 1
        
        for epoch in range(self.start_epoch, self.start_epoch + self.additional_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCA {epoch}/{final_epoch}")
            print(f"{'='*70}")
            
            # Antrenare
            train_metrics = self.train_epoch()
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1_macro']:.4f}")
            
            # Validare
            val_metrics = self.evaluate(self.val_loader, desc="Validation")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1_macro']:.4f}")
            
            # Salvare istoric
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1_macro'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            
            # Early stopping și salvare best model
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_epoch = epoch
                self.patience_counter = 0
                print(f"  ⭐ Nou BEST F1: {self.best_val_f1:.4f}")
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                print(f"  Early stopping counter: {self.patience_counter}/{self.early_stopping_patience}")
            
            # Salvare checkpoint periodic (la fiecare 5 epoci)
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⛔ Early stopping declanșat după {epoch} epoci")
                print(f"Best F1: {self.best_val_f1:.4f} la epoca {self.best_epoch}")
                break
        
        # Sfârșitul antrenării
        training_time = time.time() - start_time
        training_time_minutes = training_time / 60
        
        print(f"\n{'='*70}")
        print(f"✅ Antrenare finalizată!")
        print(f"Timp total: {training_time_minutes:.2f} minute")
        print(f"Best validation F1: {self.best_val_f1:.4f} (epoca {self.best_epoch})")
        print(f"{'='*70}\n")
        
        # Încarcă best model și evaluează pe test
        self.load_best_checkpoint()
        test_metrics = self.evaluate(self.test_loader, desc="Testing")
        test_metrics_full = self.compute_test_metrics(test_metrics)
        
        print(f"\n{'='*70}")
        print("TEST SET RESULTS (best model)")
        print(f"{'='*70}")
        print(f"Test Loss: {test_metrics_full['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics_full['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_metrics_full['f1_macro']:.4f}")
        print(f"Test F1 (weighted): {test_metrics_full['f1_weighted']:.4f}")
        print(f"Test Balanced Accuracy: {test_metrics_full['balanced_accuracy']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=['unsatisfied', 'neutral', 'satisfied']
        ))
        
        # Salvare rezultate
        self.save_test_metrics(test_metrics_full, training_time_minutes)
        self.save_history()
        self.save_loss_plot()
        self.save_config()
        
        return test_metrics_full
    
    def save_config(self):
        """Salvează configurația actualizată cu informații despre continuarea antrenării."""
        config = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'resumed_from_checkpoint': str(self.checkpoint_path),
            'start_epoch': self.start_epoch,
            'additional_epochs': self.additional_epochs,
            'final_epoch': self.start_epoch + self.additional_epochs - 1,
            'training_hyperparameters': self.hyperparameters,
            'model_architecture': self.saved_model_config,
            'scheduler_config': {
                'type': 'LinearWithWarmup',
                'num_warmup_steps': self.warmup_steps,
                'num_training_steps': self.total_steps,
                'warmup_ratio': 0.1,
            }
        }
        
        path = self.checkpoint_dir / 'training_config.json'
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Salvat configurație actualizată la {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Continuă antrenarea modelului CCMT de la un checkpoint salvat"
    )
    
    # Argumentele principale
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/ccmt_multimodal/best_model.pt',
        help='Calea către checkpoint-ul de pornire (default: best_model.pt)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Numărul de epoci suplimentare de antrenare (default: 10)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Opțional: Noul learning rate (default: păstrează din checkpoint)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size (default: 128)'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Gradient accumulation steps (default: 1)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay pentru optimizer (default: 0.01)'
    )
    parser.add_argument(
        '--keep-early-stopping-counter',
        action='store_true',
        help='Păstrează contorul early stopping din checkpoint (default: resetează)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=5,
        help='Early stopping patience (default: 5)'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='MSP_Podcast/embeddings',
        help='Directorul cu embeddings precomputate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device pentru antrenare (default: cuda dacă e disponibil)'
    )
    
    args = parser.parse_args()
    
    # Optimizări CUDA (ca în train_ccmt_multimodal.py)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True  # Auto-tune pentru performanță
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 pentru matmul
        torch.backends.cudnn.allow_tf32 = True  # TensorFloat-32 pentru cuDNN
    
    # Verificare
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint-ul nu există: {checkpoint_path}")
    
    print(f"\n{'='*70}")
    print("CONFIGURARE CONTINUARE ANTRENARE CCMT")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoci suplimentare: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Device: {args.device}")
    if args.lr:
        print(f"Noul learning rate: {args.lr}")
    if args.device == "cuda":
        print(f"CUDA optimizations: cuDNN benchmark=True, TF32 enabled")
        print(f"Mixed Precision (AMP): True")
    print(f"{'='*70}\n")
    
    # Încarcă dataset
    embeddings_dir = PROJECT_ROOT / args.embeddings_dir
    
    print("📊 Încărcare dataset cu embeddings precomputate...")
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
    test_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition='val',  # Folosim val ca test
        device='cpu'
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    print(f"\n✓ Datasets loaded:")
    print(f"  Train: {len(train_dataset)} samples, batches: {(len(train_dataset) + args.batch_size - 1) // args.batch_size}")
    print(f"  Val: {len(val_dataset)} samples, batches: {(len(val_dataset) + args.batch_size - 1) // args.batch_size}")
    print(f"  Test: {len(test_dataset)} samples, batches: {(len(test_dataset) + args.batch_size - 1) // args.batch_size}")
    
    # Creare DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if args.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if args.device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Încarcă configurația modelului din checkpoint
    config_path = checkpoint_path.parent / 'training_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        model_config = saved_config.get('model_architecture', {})
    else:
        # Configurație default
        model_config = {
            'num_classes': 3,
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
        print("⚠ Folosesc configurație model default")
    
    # Creează modelul
    print("\n🏗️  Creare model CCMT...")
    model = load_ccmt_only_model(
        num_classes=model_config['num_classes'],
        text_en_dim=model_config['text_en_dim'],
        text_es_dim=model_config['text_es_dim'],
        audio_dim=model_config['audio_dim'],
        ccmt_dim=model_config['ccmt_dim'],
        num_patches_per_modality=model_config['num_patches_per_modality'],
        ccmt_depth=model_config['ccmt_depth'],
        ccmt_heads=model_config['ccmt_heads'],
        ccmt_mlp_dim=model_config['ccmt_mlp_dim'],
        ccmt_dropout=model_config['ccmt_dropout'],
        device=args.device,
    )
    
    # Calculează class weights (cu smoothing ca în train_ccmt_multimodal.py)
    print("\nCalculând class weights cu smoothing...")
    
    # Obține toate label-urile din dataset-ul de train
    labels_all = [sample['labels'] for sample in train_dataset]
    labels_array = np.array(labels_all)
    
    # Calculează raw weights
    class_counts = np.bincount(labels_array)
    total_samples = len(labels_array)
    raw_weights = torch.tensor(
        [total_samples / (len(class_counts) * count) for count in class_counts],
        dtype=torch.float32
    )
    
    # Aplică smoothing cu radical (sqrt) - reduce penalizarea pentru clase minoritare
    class_weights = torch.sqrt(raw_weights)
    
    print(f"  Class counts: {class_counts}")
    print(f"  Raw weights: {raw_weights.numpy()}")
    print(f"  Smoothed weights (sqrt): {class_weights.numpy()}")
    
    # Inițializează trainer pentru continuare
    trainer = CCMTContinuedTrainer(
        checkpoint_path=checkpoint_path,
        additional_epochs=args.epochs,
        new_learning_rate=args.lr,
        reset_early_stopping=not args.keep_early_stopping_counter,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        checkpoint_dir=checkpoint_path.parent,  # Salvează în același director
        early_stopping_patience=args.early_stopping_patience,
        class_weights=class_weights.to(args.device),
        batch_size=args.batch_size,
        use_amp=True,  # Mixed precision training
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
    )
    
    # Începe antrenarea continuată
    test_metrics = trainer.train()
    
    print("\n" + "="*70)
    print("✅ ANTRENARE FINALIZATĂ CU SUCCES!")
    print("="*70)


if __name__ == "__main__":
    main()
