"""
Script de testare pentru modelul CCMT Multimodal.
Evaluează pe datele de validare și salvează scorurile și graficele.
"""
from pathlib import Path
from typing import Optional
import json

import torch
import sys
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.models import load_full_multimodal_model
from scripts.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset


class CCMTTester:
    """Tester pentru modelul CCMT Multimodal."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = 3
        self.id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def evaluate(self, model, val_loader: DataLoader) -> dict:
        """Evaluează modelul și calculează metrici."""
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0

        loss_fn = torch.nn.CrossEntropyLoss()
        
        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            # Unpack batch
            audio_embeddings = batch['audio_embeddings'].to(self.device)
            text_en_embeddings = batch['text_en_embeddings'].to(self.device)
            text_es_embeddings = batch['text_es_embeddings'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = model(
                audio_embeddings=audio_embeddings,
                text_en_embeddings=text_en_embeddings,
                text_es_embeddings=text_es_embeddings,
            )
            
            logits = outputs['logits']
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # Predictions
            predictions = logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # F1 per class
        f1_per_class = {}
        for label_id, label_name in self.id2label.items():
            f1_per_class[label_name] = f1_score(
                all_labels, all_predictions,
                labels=[label_id],
                average='micro',
                zero_division=0
            )

        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_per_class": {k: float(v) for k, v in f1_per_class.items()},
            "confusion_matrix": cm.tolist(),
            "avg_loss": float(total_loss / len(val_loader)),
            "num_samples": len(all_labels),
            "predictions": all_predictions.tolist(),
            "labels": all_labels.tolist(),
        }

    def plot_confusion_matrix(self, cm, output_path):
        """Plotează confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.id2label.values()),
                   yticklabels=list(self.id2label.values()))
        plt.title('Confusion Matrix - CCMT Multimodal')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved: {output_path}")

    def plot_metrics(self, results, output_path):
        """Plotează metricile principale."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # F1 per class
        f1_per_class = results['f1_per_class']
        axes[0, 0].bar(f1_per_class.keys(), f1_per_class.values(), color='steelblue')
        axes[0, 0].set_title('F1 Score per Class')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(f1_per_class.values()):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Overall metrics
        metrics_names = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']
        metrics_values = [
            results['accuracy'],
            results['f1_macro'],
            results['f1_weighted'],
            results['precision_macro'],
            results['recall_macro']
        ]
        axes[0, 1].bar(metrics_names, metrics_values, color='coral')
        axes[0, 1].set_title('Overall Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(metrics_values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        # Normalized confusion matrix
        cm_normalized = results['confusion_matrix'] / np.array(results['confusion_matrix']).sum(axis=1, keepdims=True)
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                   ax=axes[1, 0],
                   xticklabels=list(self.id2label.values()),
                   yticklabels=list(self.id2label.values()))
        axes[1, 0].set_title('Normalized Confusion Matrix')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # Loss and sample count
        info_text = f"Validation Loss: {results['avg_loss']:.4f}\n"
        info_text += f"Number of Samples: {results['num_samples']}\n"
        info_text += f"Accuracy: {results['accuracy']:.4f}\n"
        info_text += f"F1 Macro: {results['f1_macro']:.4f}"
        axes[1, 1].text(0.5, 0.5, info_text, ha='center', va='center',
                       fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Metrics plot saved: {output_path}")

    def test(
        self,
        val_dataset,
        checkpoint_dir: Path,
        batch_size: int = 32,
        output_dir: Path = Path("results/ccmt_multimodal"),
    ):
        """Testează modelul și salvează rezultatele."""
        print("="*80)
        print("Testing CCMT Multimodal Model")
        print("="*80)

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found at: {checkpoint_dir}")
        
        print(f"\nLoading model from: {checkpoint_dir}")
        model = load_full_multimodal_model(
            device=self.device,
            text_en_checkpoint=None,
            text_es_checkpoint=None,
            audio_checkpoint=None,
        )
        
        # Load CCMT weights
        model_path = checkpoint_dir / "best_model.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Model weights loaded from: {model_path}")
        else:
            print(f"⚠️  Model weights not found at: {model_path}")
        
        model = model.to(self.device)
        
        # Create DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        print(f"\nEvaluating on {len(val_dataset)} validation samples...")
        results = self.evaluate(model, val_loader)
        
        # Save results
        results_json_path = output_dir / "test_results.json"
        with open(results_json_path, 'w') as f:
            results_to_save = {k: v for k, v in results.items() if k not in ['predictions', 'labels']}
            json.dump(results_to_save, f, indent=2)
        print(f"✅ Results saved: {results_json_path}")
        
        # Create plots
        cm = np.array(results['confusion_matrix'])
        self.plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
        self.plot_metrics(results, output_dir / "metrics.png")
        
        # Display results
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Macro: {results['f1_macro']:.4f}")
        print(f"F1 Weighted: {results['f1_weighted']:.4f}")
        print(f"Precision Macro: {results['precision_macro']:.4f}")
        print(f"Recall Macro: {results['recall_macro']:.4f}")
        print(f"Avg Loss: {results['avg_loss']:.4f}")
        print(f"\nF1 per Class:")
        for label_name, f1 in results['f1_per_class'].items():
            print(f"  {label_name}: {f1:.4f}")
        print("="*80)


def main():
    """Main testing function."""
    
    # Paths
    checkpoint_dir = Path("checkpoints/ccmt_multimodal")
    embeddings_dir = Path("MSP_Podcast/embeddings")
    output_dir = Path("results/ccmt_multimodal")
    
    # Verify files
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    print("="*80)
    print("Loading Precomputed Embeddings for Multimodal Model")
    print("="*80)
    
    # Load validation dataset
    print("\nLoading Validation dataset...")
    val_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition="val",
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"   Val: {len(val_dataset)} samples\n")
    
    # Create tester
    tester = CCMTTester()
    
    # Testing
    tester.test(
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        batch_size=32,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
