from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import AutoPeftModelForSequenceClassification
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.data.audio_datasets import AudioCollator
from src.utils.peft_audio import load_peft_audio_classification_model
from src.utils.regression_trainers import (
    _prepare_regression_labels,
    _prepare_regression_logits,
    compute_regression_metrics,
    denormalize_val_arousal,
)


class TextRegressionTester:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = 2

    @torch.no_grad()
    def evaluate(self, model, val_loader: DataLoader) -> dict:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        loss_fn = nn.MSELoss()
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = _prepare_regression_labels(
                batch["labels"].to(self.device, dtype=torch.float32),
                self.num_labels,
            )

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = _prepare_regression_logits(outputs.logits, self.num_labels)
            if predictions.shape != labels.shape:
                raise ValueError(
                    f"Regression output shape mismatch: predictions {tuple(predictions.shape)} vs labels {tuple(labels.shape)}"
                )
            loss = loss_fn(predictions, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

        y_pred = denormalize_val_arousal(torch.cat(all_predictions, dim=0)).numpy()
        y_true = denormalize_val_arousal(torch.cat(all_labels, dim=0)).numpy()
        results = compute_regression_metrics(y_true, y_pred)
        results["avg_loss"] = float(total_loss / total_samples)
        results["num_samples"] = int(total_samples)
        results["predictions"] = y_pred.tolist()
        results["labels"] = y_true.tolist()
        return results

    def plot_metrics(self, results: dict, output_path: Path, title: str):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        metric_names = ["MAE", "RMSE", "R2", "CCC Mean", "Pearson Mean"]
        metric_values = [
            results["mae"],
            results["rmse"],
            results["r2"],
            results["ccc_mean"],
            results["pearson_mean"],
        ]
        axes[0, 0].bar(metric_names, metric_values, color=["#3a86ff", "#8338ec", "#ff006e", "#fb5607", "#2a9d8f"])
        axes[0, 0].set_title("Overall Regression Metrics")
        axes[0, 0].tick_params(axis="x", rotation=25)
        for index, value in enumerate(metric_values):
            axes[0, 0].text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

        target_names = ["Valence", "Arousal"]
        ccc_values = [results["ccc_valence"], results["ccc_arousal"]]
        axes[0, 1].bar(target_names, ccc_values, color=["#457b9d", "#e76f51"])
        axes[0, 1].set_title("CCC per Target")
        axes[0, 1].set_ylim(min(-0.1, min(ccc_values) - 0.05), 1.0)

        predictions = np.array(results["predictions"])
        labels = np.array(results["labels"])
        axes[1, 0].scatter(labels[:, 0], predictions[:, 0], alpha=0.5, label="Valence", color="#3a86ff")
        axes[1, 0].scatter(labels[:, 1], predictions[:, 1], alpha=0.5, label="Arousal", color="#ef476f")
        axes[1, 0].plot([1, 7], [1, 7], linestyle="--", color="black", linewidth=1)
        axes[1, 0].set_title("Predicted vs True")
        axes[1, 0].set_xlabel("True")
        axes[1, 0].set_ylabel("Predicted")
        axes[1, 0].legend()

        info_text = (
            f"Loss: {results['avg_loss']:.4f}\n"
            f"Samples: {results['num_samples']}\n"
            f"MAE Valence: {results['mae_valence']:.4f}\n"
            f"MAE Arousal: {results['mae_arousal']:.4f}\n"
            f"Pearson Valence: {results['pearson_valence']:.4f}\n"
            f"Pearson Arousal: {results['pearson_arousal']:.4f}"
        )
        axes[1, 1].text(0.5, 0.5, info_text, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="#f1faee", alpha=0.8))
        axes[1, 1].axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def test(self, val_dataset, checkpoint_dir: Path, batch_size: int, output_dir: Path, title: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / "best_model"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Model not found at: {best_model_path}")

        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            best_model_path,
            num_labels=self.num_labels,
            problem_type="regression",
            ignore_mismatched_sizes=True,
        ).to(self.device)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=data_collator)
        results = self.evaluate(model, val_loader)

        results_json_path = output_dir / "test_results.json"
        with open(results_json_path, "w", encoding="utf-8") as file_handle:
            json.dump({key: value for key, value in results.items() if key not in ["predictions", "labels"]}, file_handle, indent=2)

        self.plot_metrics(results, output_dir / "metrics.png", title)
        return results


class AudioRegressionTester:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def evaluate(self, model, val_loader: DataLoader) -> dict:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        loss_fn = nn.MSELoss()
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device, dtype=torch.float32)

            outputs = model(input_values=input_values, attention_mask=attention_mask)
            predictions = outputs.logits.float().view(-1, 2)
            labels = labels.view(-1, 2)
            loss = loss_fn(predictions, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

        y_pred = denormalize_val_arousal(torch.cat(all_predictions, dim=0)).numpy()
        y_true = denormalize_val_arousal(torch.cat(all_labels, dim=0)).numpy()
        results = compute_regression_metrics(y_true, y_pred)
        results["avg_loss"] = float(total_loss / total_samples)
        results["num_samples"] = int(total_samples)
        results["predictions"] = y_pred.tolist()
        results["labels"] = y_true.tolist()
        return results

    def plot_metrics(self, results: dict, output_path: Path, title: str):
        TextRegressionTester().plot_metrics(results, output_path, title)

    def test(self, val_dataset, checkpoint_dir: Path, batch_size: int, output_dir: Path, title: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / "best_model"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Model not found at: {best_model_path}")

        model = load_peft_audio_classification_model(
            best_model_path,
            num_labels=2,
            problem_type="regression",
        ).to(self.device)
        collate_fn = AudioCollator()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        results = self.evaluate(model, val_loader)

        results_json_path = output_dir / "test_results.json"
        with open(results_json_path, "w", encoding="utf-8") as file_handle:
            json.dump({key: value for key, value in results.items() if key not in ["predictions", "labels"]}, file_handle, indent=2)

        self.plot_metrics(results, output_dir / "metrics.png", title)
        return results