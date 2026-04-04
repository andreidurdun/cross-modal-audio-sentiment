"""
Script de testare pentru modelul RoBERTa Text ES.
Evaluează pe datele de validare și salvează scorurile și graficele.
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from pathlib import Path
from typing import Optional
import json
from datetime import datetime

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

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from peft import AutoPeftModelForSequenceClassification

from src.data.dataset import MSP_Podcast_Dataset
from src.data.text_datasets import TextEncoderDataset
from src.utils.metrics import compute_classification_metrics


class TextEsTester:
    """Tester pentru modelul RoBERTa Text ES."""

    def __init__(self, model_name: str = "roberta-base", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
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

        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        return compute_classification_metrics(
            labels=all_labels,
            predictions=all_predictions,
            id2label=self.id2label,
            total_loss=total_loss,
            num_batches=len(val_loader),
        )

    def plot_confusion_matrix(self, cm, output_path):
        """Plotează confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.id2label.values()),
                   yticklabels=list(self.id2label.values()))
        plt.title('Confusion Matrix - RoBERTa Text ES')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved: {output_path}")

    def plot_metrics(self, results, output_path):
        """Plotează metricile principale."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # F1 per clase
        f1_per_class = results['f1_per_class']
        axes[0, 0].bar(f1_per_class.keys(), f1_per_class.values(), color='steelblue')
        axes[0, 0].set_title('F1 Score per Class')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(f1_per_class.values()):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Accuracy vs F1
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
        
        # Normalizată confusion matrix
        cm_normalized = results['confusion_matrix'] / np.array(results['confusion_matrix']).sum(axis=1, keepdims=True)
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                   ax=axes[1, 0],
                   xticklabels=list(self.id2label.values()),
                   yticklabels=list(self.id2label.values()))
        axes[1, 0].set_title('Normalized Confusion Matrix')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # Loss și sample count
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
        output_dir: Path = Path("results/roberta_text_es"),
    ):
        """Testează modelul și salvează rezultatele."""
        print("="*80)
        print("Testing RoBERTa Text ES Model")
        print("="*80)

        # Creează directoare output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Încarc modelul
        checkpoint_dir = Path(checkpoint_dir)
        best_model_path = checkpoint_dir / "best_model"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Model not found at: {best_model_path}")
        
        print(f"\nLoading model from: {best_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        model = AutoPeftModelForSequenceClassification.from_pretrained(best_model_path)
        model = model.to(self.device)
        
        # DataLoader
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=data_collator,
        )
        
        print(f"\nEvaluating on {len(val_dataset)} validation samples...")
        results = self.evaluate(model, val_loader)
        
        # Salvează rezultatele
        results_json_path = output_dir / "test_results.json"
        with open(results_json_path, 'w') as f:
            results_to_save = {k: v for k, v in results.items() if k not in ['predictions', 'labels']}
            json.dump(results_to_save, f, indent=2)
        print(f"✅ Results saved: {results_json_path}")
        
        # Crează grafice
        cm = np.array(results['confusion_matrix'])
        self.plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
        self.plot_metrics(results, output_dir / "metrics.png")
        
        # Afișează rezultatele
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
    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_es_json = data_dir / "Transcription_es.json"
    checkpoint_dir = Path("checkpoints/roberta_text_es")
    output_dir = Path("results/roberta_text_es")
    
    # Verificare fișiere
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_es_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_es_json}")
    
    print("="*80)
    print("Loading MSP-Podcast Spanish Text Data")
    print("="*80)
    
    # Load validation dataset
    print("\nLoading Validation dataset...")
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_es_json=str(transcripts_es_json),
        partition="Development",
        modalities=['text_es'],
        use_cache=True,
        max_workers=8
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"   Val: {len(val_dataset_msp)} samples\n")
    
    # Create tester and wrap dataset
    tester = TextEsTester()
    best_model_path = checkpoint_dir / "best_model"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model not found at: {best_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    val_dataset = TextEncoderDataset(
        val_dataset_msp,
        tokenizer,
        text_fields=["text_es"],
        max_length=512,
        padding=True,
        sanitize_input_ids=True,
    )
    
    # Testing
    tester.test(
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        batch_size=64,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
