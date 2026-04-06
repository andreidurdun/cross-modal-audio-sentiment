from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.data.audio_datasets import AudioCollator


def denormalize_val_arousal(targets: torch.Tensor) -> torch.Tensor:
    return targets * 6.0 + 1.0


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mean_true = np.mean(y_true, axis=0)
    mean_pred = np.mean(y_pred, axis=0)
    var_true = np.var(y_true, axis=0)
    var_pred = np.var(y_pred, axis=0)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred), axis=0)
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return {
        "valence": float(ccc[0]),
        "arousal": float(ccc[1]),
        "mean": float(np.mean(ccc)),
    }


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    results = {}
    for index, name in enumerate(("valence", "arousal")):
        if np.std(y_true[:, index]) < 1e-8 or np.std(y_pred[:, index]) < 1e-8:
            results[name] = 0.0
        else:
            corr = np.corrcoef(y_true[:, index], y_pred[:, index])[0, 1]
            results[name] = float(corr)
    results["mean"] = float((results["valence"] + results["arousal"]) / 2.0)
    return results


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    pearson = safe_pearsonr(y_true, y_pred)
    mse_per_target = np.mean((y_true - y_pred) ** 2, axis=0)
    mae_per_target = np.mean(np.abs(y_true - y_pred), axis=0)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mse_valence": float(mse_per_target[0]),
        "mse_arousal": float(mse_per_target[1]),
        "mae_valence": float(mae_per_target[0]),
        "mae_arousal": float(mae_per_target[1]),
        "ccc_valence": ccc["valence"],
        "ccc_arousal": ccc["arousal"],
        "ccc_mean": ccc["mean"],
        "pearson_valence": pearson["valence"],
        "pearson_arousal": pearson["arousal"],
        "pearson_mean": pearson["mean"],
    }


def _get_linear_output_dim(layer: nn.Linear) -> int:
    weight = getattr(layer, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
        return int(weight.shape[0])
    return int(layer.out_features)


def _prepare_regression_labels(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    if labels.ndim == 1:
        if labels.numel() % num_labels != 0:
            raise ValueError(
                f"Regression labels are 1D with {labels.numel()} values; cannot reshape into target width {num_labels}."
            )
        return labels.reshape(-1, num_labels)

    if labels.ndim != 2 or labels.shape[-1] != num_labels:
        raise ValueError(
            f"Regression labels shape mismatch: got {tuple(labels.shape)}, expected (*, {num_labels})."
        )

    return labels.reshape(-1, num_labels)


def _prepare_regression_logits(logits: torch.Tensor, num_labels: int) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != num_labels:
        raise ValueError(
            f"Regression logits shape mismatch: got {tuple(logits.shape)}, expected (*, {num_labels})."
        )

    return logits.float()


class TextRegressionTrainer:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = 2
        self.id2label = {0: "valence", 1: "arousal"}
        self.label2id = {value: key for key, value in self.id2label.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _replace_linear_layer(self, layer: nn.Linear) -> nn.Linear:
        new_layer = nn.Linear(
            in_features=layer.in_features,
            out_features=self.num_labels,
            bias=layer.bias is not None,
        )
        nn.init.normal_(new_layer.weight, mean=0.0, std=0.02)
        if new_layer.bias is not None:
            nn.init.zeros_(new_layer.bias)
        return new_layer.to(device=layer.weight.device, dtype=layer.weight.dtype)

    def _replace_nested_linear_layer(self, module: nn.Module, attribute_name: str) -> bool:
        layer = getattr(module, attribute_name, None)
        if not isinstance(layer, nn.Linear):
            return False
        if _get_linear_output_dim(layer) == self.num_labels:
            return False
        setattr(module, attribute_name, self._replace_linear_layer(layer))
        return True

    def _iter_wrapped_modules(self, module: nn.Module) -> list[nn.Module]:
        modules = [module]

        original_module = getattr(module, "original_module", None)
        if isinstance(original_module, nn.Module):
            modules.append(original_module)

        modules_to_save = getattr(module, "modules_to_save", None)
        if isinstance(modules_to_save, nn.ModuleDict):
            modules.extend(list(modules_to_save.values()))

        return modules

    def _ensure_regression_head_shape(self, model):
        head_replaced = False

        classifier = getattr(model, "classifier", None)
        if isinstance(classifier, nn.Linear):
            if _get_linear_output_dim(classifier) != self.num_labels:
                model.classifier = self._replace_linear_layer(classifier)
                head_replaced = True
        elif classifier is not None:
            for classifier_module in self._iter_wrapped_modules(classifier):
                head_replaced = self._replace_nested_linear_layer(classifier_module, "out_proj") or head_replaced

        score = getattr(model, "score", None)
        if isinstance(score, nn.Linear) and _get_linear_output_dim(score) != self.num_labels:
            model.score = self._replace_linear_layer(score)
            head_replaced = True
        elif score is not None:
            for score_module in self._iter_wrapped_modules(score):
                if isinstance(score_module, nn.Linear) and _get_linear_output_dim(score_module) != self.num_labels:
                    replacement = self._replace_linear_layer(score_module)
                    if score_module is score:
                        model.score = replacement
                    head_replaced = True

        if not head_replaced:
            output_dim = None
            classifier = getattr(model, "classifier", None)
            if isinstance(classifier, nn.Linear):
                output_dim = _get_linear_output_dim(classifier)
            elif classifier is not None:
                for classifier_module in self._iter_wrapped_modules(classifier):
                    out_proj = getattr(classifier_module, "out_proj", None)
                    if isinstance(out_proj, nn.Linear):
                        output_dim = _get_linear_output_dim(out_proj)
                        break
            elif isinstance(getattr(model, "score", None), nn.Linear):
                output_dim = _get_linear_output_dim(model.score)

            if output_dim is not None and output_dim != self.num_labels:
                raise ValueError(
                    f"Unsupported classification head output size {output_dim}; expected {self.num_labels}."
                )

        model.num_labels = self.num_labels
        model.config.num_labels = self.num_labels
        model.config.id2label = dict(self.id2label)
        model.config.label2id = dict(self.label2id)
        model.config.problem_type = "regression"
        return model

    def setup_model_with_lora(self, lora_r: int = 16, lora_alpha: int = 32):
        print("Configuring 4-bit Quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["classifier"],
        )

        print("Loading base model in 4-bit...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="regression",
            ignore_mismatched_sizes=True,
            quantization_config=bnb_config,
        )

        model = self._ensure_regression_head_shape(model)
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "value", "key", "output.dense", "intermediate.dense"],
            modules_to_save=["classifier"],
        )

        model = get_peft_model(model, lora_config)
        model = self._ensure_regression_head_shape(model)
        model.print_trainable_parameters()
        return model.to(self.device)

    def train_epoch(self, model, train_loader: DataLoader, optimizer, scheduler, gradient_accumulation_steps: int = 1) -> float:
        model.train()
        total_loss = 0.0
        loss_fn = nn.MSELoss()
        progress_bar = tqdm(train_loader, desc="Training", position=0, leave=True, dynamic_ncols=True, mininterval=0.5)
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
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
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            actual_loss = loss.item() * gradient_accumulation_steps
            total_loss += actual_loss
            progress_bar.set_postfix({"loss": f"{actual_loss:.4f}"})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, model, val_loader: DataLoader) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        loss_fn = nn.MSELoss()

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
        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["loss"] = float(total_loss / total_samples)
        return metrics

    def _plot_training_curves(self, train_losses, val_losses, checkpoint_dir: Path):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2, marker="o")
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2, marker="s")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = checkpoint_dir / "training_curves.pdf"
        plt.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to: {plot_path}")

    def train(
        self,
        train_dataset,
        val_dataset,
        checkpoint_dir: Path,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        lora_r: int,
        lora_alpha: int,
        gradient_accumulation_steps: int,
    ):
        print("=" * 80)
        print("Text Regression Training with QLoRA")
        print("=" * 80)

        start_time = time.time()
        model = self.setup_model_with_lora(lora_r=lora_r, lora_alpha=lora_alpha)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=data_collator)

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        best_val_ccc = float("-inf")
        best_metrics: dict[str, float] = {}
        last_train_loss = 0.0
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_losses = []
        val_losses = []

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size} (Effective: {batch_size * gradient_accumulation_steps})")
        print(f"Learning rate: {learning_rate}")
        print(f"LoRA r={lora_r}, alpha={lora_alpha}")
        print("Best model saved by: CCC Mean\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)

            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, gradient_accumulation_steps)
            train_losses.append(train_loss)
            last_train_loss = train_loss
            print(f"Train Loss: {train_loss:.4f}")

            val_metrics = self.evaluate(model, val_loader)
            val_losses.append(val_metrics["loss"])
            print(
                f"Val Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f} | "
                f"R2: {val_metrics['r2']:.4f} | CCC Mean: {val_metrics['ccc_mean']:.4f}"
            )

            if val_metrics["ccc_mean"] > best_val_ccc:
                best_val_ccc = val_metrics["ccc_mean"]
                best_metrics = dict(val_metrics)
                best_model_path = checkpoint_dir / "best_model"
                print(f"New best model! CCC Mean: {best_val_ccc:.4f} - Saving to {best_model_path}")
                model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)

        total_time_seconds = time.time() - start_time
        training_results = {
            "total_training_time": {
                "seconds": total_time_seconds,
                "minutes": total_time_seconds / 60.0,
                "hours": total_time_seconds / 3600.0,
            },
            "best_model_metrics": best_metrics,
            "final_train_metrics": {
                "train_loss": float(last_train_loss),
            },
            "hyperparameters": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            "training_curves": {
                "train_losses": [float(loss) for loss in train_losses],
                "val_losses": [float(loss) for loss in val_losses],
            },
        }

        results_file = checkpoint_dir / "training_results.json"
        with open(results_file, "w", encoding="utf-8") as file_handle:
            json.dump(training_results, file_handle, indent=2)
        print(f"\nTraining results saved to: {results_file}")
        self._plot_training_curves(train_losses, val_losses, checkpoint_dir)


class AudioRegressionTrainer:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = 2
        self.id2label = {0: "valence", 1: "arousal"}
        self.label2id = {value: key for key, value in self.id2label.items()}
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def _ensure_regression_head_shape(self, model):
        classifier = getattr(model, "classifier", None)
        if isinstance(classifier, nn.Linear) and _get_linear_output_dim(classifier) != self.num_labels:
            model.classifier = nn.Linear(
                in_features=classifier.in_features,
                out_features=self.num_labels,
                bias=classifier.bias is not None,
            ).to(device=classifier.weight.device, dtype=classifier.weight.dtype)
        elif classifier is not None and hasattr(classifier, "out_features") and classifier.out_features != self.num_labels:
            raise ValueError(
                f"Unsupported audio classification head output size {classifier.out_features}; expected {self.num_labels}."
            )

        model.num_labels = self.num_labels
        model.config.num_labels = self.num_labels
        model.config.id2label = dict(self.id2label)
        model.config.label2id = dict(self.label2id)
        model.config.problem_type = "regression"
        return model

    def setup_model_with_lora(self, lora_r: int = 16, lora_alpha: int = 32):
        print("Loading base model in native precision...")
        model = AutoModelForAudioClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="regression",
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
        )

        model = self._ensure_regression_head_shape(model)
        model.config.use_cache = False
        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear",
            modules_to_save=["classifier", "projector"],
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model.to(self.device)

    def train_epoch(self, model, train_loader, optimizer, scheduler, scaler, use_amp, gradient_accumulation_steps=1) -> float:
        model.train()
        total_loss = 0.0
        valid_steps = 0
        loss_fn = nn.MSELoss()
        progress_bar = tqdm(train_loader, desc="Training", position=0, leave=True, dynamic_ncols=True, mininterval=0.5)
        optimizer.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for step, batch in enumerate(progress_bar):
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, dtype=torch.float32, non_blocking=True)

            autocast_ctx = torch.amp.autocast(dtype=amp_dtype, device_type="cuda") if use_amp else nullcontext()
            with autocast_ctx:
                outputs = model(input_values=input_values, attention_mask=attention_mask)
                logits = _prepare_regression_logits(outputs.logits, self.num_labels)
                labels_view = _prepare_regression_labels(labels, self.num_labels)
                if logits.shape != labels_view.shape:
                    raise ValueError(
                        f"Regression output shape mismatch: predictions {tuple(logits.shape)} vs labels {tuple(labels_view.shape)}"
                    )
                loss = loss_fn(logits, labels_view)

            loss = loss / gradient_accumulation_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            actual_loss = loss.item() * gradient_accumulation_steps
            total_loss += actual_loss
            valid_steps += 1
            progress_bar.set_postfix({"loss": f"{actual_loss:.4f}"})

        return total_loss / max(valid_steps, 1)

    @torch.no_grad()
    def evaluate(self, model, val_loader, use_amp: bool) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        loss_fn = nn.MSELoss()
        progress_bar = tqdm(val_loader, desc="Evaluating", position=0, leave=False)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for batch in progress_bar:
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, dtype=torch.float32, non_blocking=True)

            autocast_ctx = torch.amp.autocast(dtype=amp_dtype, device_type="cuda") if use_amp else nullcontext()
            with autocast_ctx:
                outputs = model(input_values=input_values, attention_mask=attention_mask)
                logits = _prepare_regression_logits(outputs.logits, self.num_labels)
                labels_view = _prepare_regression_labels(labels, self.num_labels)
                if logits.shape != labels_view.shape:
                    raise ValueError(
                        f"Regression output shape mismatch: predictions {tuple(logits.shape)} vs labels {tuple(labels_view.shape)}"
                    )
                loss = loss_fn(logits, labels_view)

            batch_size = labels_view.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_predictions.append(logits.cpu())
            all_labels.append(labels_view.cpu())

        y_pred = denormalize_val_arousal(torch.cat(all_predictions, dim=0)).numpy()
        y_true = denormalize_val_arousal(torch.cat(all_labels, dim=0)).numpy()
        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["loss"] = float(total_loss / total_samples)
        return metrics

    def _plot_training_curves(self, train_losses, val_losses, checkpoint_dir: Path):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2, marker="o")
        plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2, marker="s")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = checkpoint_dir / "training_curves.pdf"
        plt.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to: {plot_path}")

    def train(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        checkpoint_dir: Path,
        num_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        gradient_accumulation_steps: int = 8,
        use_amp: bool = True,
        num_workers: int = 4,
    ):
        print("=" * 80)
        print("Audio Regression Training - WavLM")
        print("=" * 80)
        start_time = time.time()

        model = self.setup_model_with_lora(lora_r=lora_r, lora_alpha=lora_alpha)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            use_amp = False

        collator = AudioCollator()
        effective_num_workers = 0 if os.name == "nt" else max(0, num_workers)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=effective_num_workers, pin_memory=self.device == "cuda")
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator, num_workers=effective_num_workers, pin_memory=self.device == "cuda")
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator, num_workers=effective_num_workers, pin_memory=self.device == "cuda")

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_val_ccc = float("-inf")
        best_metrics: dict[str, float] = {}
        last_train_loss = 0.0
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_losses = []
        val_losses = []

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size} (Effective: {batch_size * gradient_accumulation_steps})")
        print(f"Learning rate: {learning_rate}")
        print(f"LoRA r={lora_r}, alpha={lora_alpha}")
        print("Best model saved by: CCC Mean\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)

            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, scaler, use_amp, gradient_accumulation_steps)
            train_losses.append(train_loss)
            last_train_loss = train_loss
            print(f"Train Loss: {train_loss:.4f}")

            val_metrics = self.evaluate(model, val_loader, use_amp)
            val_losses.append(val_metrics["loss"])
            print(
                f"Val Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f} | "
                f"R2: {val_metrics['r2']:.4f} | CCC Mean: {val_metrics['ccc_mean']:.4f}"
            )

            if val_metrics["ccc_mean"] > best_val_ccc:
                best_val_ccc = val_metrics["ccc_mean"]
                best_metrics = dict(val_metrics)
                best_model_path = checkpoint_dir / "best_model"
                print(f"New best model! CCC Mean: {best_val_ccc:.4f} - Saving...")

                try:
                    merged_model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
                    merged_model.save_pretrained(best_model_path)
                    print("Saved merged full model for downstream backbone loading.")
                except Exception as exc:
                    print(f"[WARN] Could not merge LoRA adapters ({exc}). Saving adapter-only checkpoint.")
                    model.save_pretrained(best_model_path)

                self.feature_extractor.save_pretrained(best_model_path)

        total_time_seconds = time.time() - start_time
        training_results = {
            "total_training_time": {
                "seconds": total_time_seconds,
                "minutes": total_time_seconds / 60.0,
                "hours": total_time_seconds / 3600.0,
            },
            "best_model_metrics": best_metrics,
            "final_train_metrics": {
                "train_loss": float(last_train_loss),
            },
            "hyperparameters": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "use_amp": use_amp,
            },
            "training_curves": {
                "train_losses": [float(loss) for loss in train_losses],
                "val_losses": [float(loss) for loss in val_losses],
            },
        }

        results_file = checkpoint_dir / "training_results.json"
        with open(results_file, "w", encoding="utf-8") as file_handle:
            json.dump(training_results, file_handle, indent=2)
        print(f"\nTraining results saved to: {results_file}")
        self._plot_training_curves(train_losses, val_losses, checkpoint_dir)

        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_metrics = self.evaluate(model, test_loader, use_amp)
            print(
                f"Test Loss: {test_metrics['loss']:.4f} | MAE: {test_metrics['mae']:.4f} | "
                f"R2: {test_metrics['r2']:.4f} | CCC Mean: {test_metrics['ccc_mean']:.4f}"
            )