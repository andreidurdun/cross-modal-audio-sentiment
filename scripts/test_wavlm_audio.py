"""
Script de testare pentru modelul WavLM Audio.
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

from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)
from peft import AutoPeftModelForAudioClassification

from src.data.dataset import MSP_Podcast_Dataset


class AudioWaveLMDataset(Dataset):
    """Dataset wrapper pentru audio WavLM."""

    def __init__(self, msp_dataset: MSP_Podcast_Dataset, feature_extractor):
        self.msp_dataset = msp_dataset
        self.feature_extractor = feature_extractor
        self.sample_rate = 16000

    def __len__(self):
        return len(self.msp_dataset)

    def __getitem__(self, idx):
        sample = self.msp_dataset[idx]
        audio = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
        
        # Resample la 16kHz dacă e necesar
        if sample.get('sample_rate', 16000) != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample.get('sample_rate', 16000), target_sr=self.sample_rate)
        
        # Feature extraction
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160000  # ~10 sec at 16kHz
        )
        
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs.get("attention_mask", torch.ones(inputs["input_values"].shape[-1])).squeeze(0) if "attention_mask" in inputs else torch.ones(inputs["input_values"].shape[-1]),
            "labels": torch.tensor(int(sample['label_id']), dtype=torch.long),
        }


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
    def evaluate(self, model, val_loader: DataLoader) -> dict:
        """Evaluează modelul și calculează metrici."""
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
                labels=labels,
            )

            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calcul metrici
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        # F1 per clase
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
        plt.title('Confusion Matrix - WavLM Audio')
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
        batch_size: int = 8,
        output_dir: Path = Path("results/wavlm_audio"),
    ):
        """Testează modelul și salvează rezultatele."""
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
        model = AutoPeftModelForAudioClassification.from_pretrained(best_model_path)
        model = model.to(self.device)
        
        # Create DataLoader with custom collate function
        def collate_fn(batch):
            """Custom collate function for audio data."""
            max_length = max(sample["input_values"].shape[0] for sample in batch)
            
            input_values = []
            attention_masks = []
            labels = []
            
            for sample in batch:
                # Pad input values
                pad_length = max_length - sample["input_values"].shape[0]
                padded = torch.nn.functional.pad(sample["input_values"], (0, pad_length))
                input_values.append(padded)
                
                # Create attention mask
                attention_mask = torch.ones(max_length)
                if pad_length > 0:
                    attention_mask[-pad_length:] = 0
                attention_masks.append(attention_mask)
                
                labels.append(sample["labels"])
            
            return {
                "input_values": torch.stack(input_values),
                "attention_mask": torch.stack(attention_masks),
                "labels": torch.stack(labels),
            }
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
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
    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    audio_dir = data_dir / "Audios"
    checkpoint_dir = Path("checkpoints/wavlm_audio")
    output_dir = Path("results/wavlm_audio")
    
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
        partition="Development",
        modalities=['audio'],
        use_cache=True,
        max_workers=8
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"   Val: {len(val_dataset_msp)} samples\n")
    
    # Create tester and wrap dataset
    tester = AudioTester()
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    val_dataset = AudioWaveLMDataset(val_dataset_msp, feature_extractor)
    
    # Testing
    tester.test(
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        batch_size=8,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
