"""
Script de testare pentru modelul RoBERTa Text FR.
Evalueaza pe datele de validare/test si salveaza scorurile si graficele.
"""

from pathlib import Path
from typing import Optional
import argparse
import json

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
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


class TextFrTester:
    """Tester pentru modelul RoBERTa Text FR."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = 3
        self.id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def evaluate(self, model, val_loader: DataLoader) -> dict:
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0

        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            all_predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return compute_classification_metrics(
            labels=np.array(all_labels),
            predictions=np.array(all_predictions),
            id2label=self.id2label,
            total_loss=total_loss,
            num_batches=len(val_loader),
        )

    def plot_confusion_matrix(self, cm, output_path):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(self.id2label.values()),
                    yticklabels=list(self.id2label.values()))
        plt.title("Confusion Matrix - RoBERTa Text FR")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Confusion matrix saved: {output_path}")

    def test(
        self,
        val_dataset,
        checkpoint_dir: Path,
        batch_size: int = 32,
        output_dir: Path = Path("results/roberta_text_fr"),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        best_model_path = Path(checkpoint_dir) / "best_model"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Model not found at: {best_model_path}")

        print(f"\nLoading model from: {best_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        model = AutoPeftModelForSequenceClassification.from_pretrained(best_model_path, num_labels=3).to(self.device)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        print(f"\nEvaluating on {len(val_dataset)} samples...")
        results = self.evaluate(model, val_loader)

        results_json_path = output_dir / "test_results.json"
        with open(results_json_path, "w") as f:
            json.dump({k: v for k, v in results.items() if k not in ("predictions", "labels")}, f, indent=2)
        print(f"[OK] Results saved: {results_json_path}")

        self.plot_confusion_matrix(np.array(results["confusion_matrix"]), output_dir / "confusion_matrix.png")

        print("\n" + "=" * 80)
        print("TEST RESULTS - RoBERTa Text FR")
        print("=" * 80)
        print(f"Accuracy:         {results['accuracy']:.4f}")
        print(f"F1 Macro:         {results['f1_macro']:.4f}")
        print(f"F1 Weighted:      {results['f1_weighted']:.4f}")
        print(f"Precision Macro:  {results['precision_macro']:.4f}")
        print(f"Recall Macro:     {results['recall_macro']:.4f}")
        print(f"Avg Loss:         {results['avg_loss']:.4f}")
        print("\nF1 per Class:")
        for label_name, f1 in results["f1_per_class"].items():
            print(f"  {label_name}: {f1:.4f}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Testeaza un checkpoint RoBERTa Text FR")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/roberta_text_fr"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/roberta_text_fr"))
    parser.add_argument("--transcript-json", type=Path, default=None, help="Calea catre Transcription_fr JSON. Implicit: MSP_Podcast/Transcription_fr.json")
    parser.add_argument("--partition", type=str, default="Development", help="Partitia de evaluat: Development, Test1, Train")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_fr_json = args.transcript_json or (data_dir / "Transcription_fr.json")
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_fr_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_fr_json}")

    print("=" * 80)
    print("Loading MSP-Podcast French Text Data")
    print("=" * 80)

    print(f"\nLoading {args.partition} dataset...")
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_fr_json=str(transcripts_fr_json),
        partition=args.partition,
        modalities=["text_fr"],
    )

    print(f"[OK] Data loaded: {len(val_dataset_msp)} samples")

    tester = TextFrTester()
    best_model_path = checkpoint_dir / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    val_dataset = TextEncoderDataset(
        val_dataset_msp,
        tokenizer,
        text_fields=["text_fr"],
        max_length=512,
        padding=True,
        sanitize_input_ids=True,
    )

    tester.test(
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        batch_size=args.batch_size,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
