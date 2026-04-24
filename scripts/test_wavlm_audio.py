"""
Script de testare pentru modelul WavLM Audio.
Evalueaza pe datele de validare si salveaza scorurile si graficele.
"""
import argparse
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

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)

from src.data.dataset import MSP_Podcast_Dataset
from src.data.audio_datasets import AudioWaveLMDataset, AudioCollator
from src.utils.metrics import compute_classification_metrics
from src.utils.peft_audio import load_peft_audio_classification_model


class AudioTester:
    """Tester pentru modelul WavLM Audio."""

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = 3
        self.id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def evaluate(self, model, val_loader: DataLoader, loss_fn: Optional[torch.nn.Module] = None) -> dict:
        """Evalueaza modelul si calculeaza metrici."""
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0

        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
            )

            logits = outputs.logits.float().view(-1, self.num_labels)
            labels_flat = labels.view(-1)
            batch_loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
            total_loss += batch_loss_fn(logits, labels_flat).item()
            predictions = logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

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
        """Ploteaza confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.id2label.values()),
                   yticklabels=list(self.id2label.values()))
        plt.title('Confusion Matrix - WavLM Audio')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Confusion matrix saved: {output_path}")

    def plot_metrics(self, results, output_path):
        """Ploteaza metricile principale."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # F1 per clase
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
        
        # Loss si sample count
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
        print(f"[OK] Metrics plot saved: {output_path}")

    def test(
        self,
        val_dataset,
        checkpoint_dir: Path,
        batch_size: int = 8,
        output_dir: Path = Path("results/wavlm_audio"),
        loss_fn: Optional[torch.nn.Module] = None,
    ):
        """Testeaza modelul si salveaza rezultatele."""
        print("="*80)
        print("Testing WavLM Audio Model")
        print("="*80)

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        checkpoint_dir = Path(checkpoint_dir)
        best_model_path = checkpoint_dir / "best_model"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Model not found at: {best_model_path}")
        
        print(f"\nLoading model from: {best_model_path}")
        model = load_peft_audio_classification_model(best_model_path, num_labels=self.num_labels)
        model = model.to(self.device)
        
        collate_fn = AudioCollator()

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        print(f"\nEvaluating on {len(val_dataset)} validation samples...")
        results = self.evaluate(model, val_loader, loss_fn=loss_fn)
        
        # Save results
        results_json_path = output_dir / "test_results.json"
        with open(results_json_path, 'w') as f:
            results_to_save = {k: v for k, v in results.items() if k not in ['predictions', 'labels']}
            json.dump(results_to_save, f, indent=2)
        print(f"[OK] Results saved: {results_json_path}")
        
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

    parser = argparse.ArgumentParser(description="Testeaza un checkpoint WavLM audio sau audio KD")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/wavlm_audio"),
        help="Directorul checkpoint-ului care contine best_model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/wavlm_audio"),
        help="Directorul in care se salveaza rezultatele testarii",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("MSP_Podcast"),
        help="Directorul dataset-ului MSP_Podcast",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size pentru evaluare",
    )
    parser.add_argument(
        "--eval-profile",
        type=str,
        choices=["standard", "kd-validation"],
        default="standard",
        help="Profil de evaluare: standard pentru test generic, kd-validation pentru a replica validarea din training KD",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="Development",
        help="Partitia de evaluat: Development, Test1, Train",
    )
    args = parser.parse_args()

    # Paths
    data_dir = args.data_dir
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    audio_dir = data_dir / "Audios"
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    
    # Verify files
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    print("="*80)
    print("Loading MSP-Podcast Audio Data")
    print("="*80)
    
    # Load validation dataset
    print("\nLoading Validation dataset...")
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(audio_dir),
        labels_csv=str(labels_csv),
        partition=args.partition,
        modalities=['audio'],
    )
    
    print(f"[OK] Data loaded successfully!")
    print(f"   Val: {len(val_dataset_msp)} samples\n")
    
    # Create tester and wrap dataset
    tester = AudioTester()
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")

    if args.eval_profile == "kd-validation":
        dataset_kwargs = {
            "max_seconds": 5,
            "do_resample": False,
            "label_key": "label_id",
            "include_attention_mask": True,
            "extractor_padding": False,
            "extractor_truncation": False,
            "extractor_max_length": None,
        }

        class_weights_dataset = MSP_Podcast_Dataset(
            audio_root=str(audio_dir),
            labels_csv=str(labels_csv),
            partition="Train",
            modalities=[],
        )
        all_train_labels = class_weights_dataset.metadata["label_id"].astype(int).to_numpy()
        raw_weights = class_weights_dataset.get_class_weights(all_train_labels, device=tester.device)
        loss_fn: Optional[torch.nn.Module] = torch.nn.CrossEntropyLoss(weight=torch.sqrt(raw_weights))
    else:
        dataset_kwargs = {
            "max_seconds": 5,
            "do_resample": False,
            "label_key": "label",
            "include_attention_mask": False,
            "extractor_padding": False,
            "extractor_truncation": False,
            "extractor_max_length": None,
        }
        loss_fn = None

    val_dataset = AudioWaveLMDataset(
        val_dataset_msp,
        feature_extractor,
        **dataset_kwargs,
    )
    
    # Testing
    tester.test(
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        batch_size=args.batch_size,
        output_dir=output_dir,
        loss_fn=loss_fn,
    )


if __name__ == "__main__":
    main()
