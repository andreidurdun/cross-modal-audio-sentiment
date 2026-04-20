from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

warnings.filterwarnings("ignore")

from src.data.audio_datasets import AudioCollator, AudioWaveLMDataset
from src.data.dataset import MSP_Podcast_Dataset
from src.data.distillation_dataset import AudioTeacherDistillationDataset
from src.data.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset
from src.models import load_ccmt_only_model
from src.utils.config import get_training_config


SUPPORTED_MODALITIES = ["text_en", "text_es", "text_de", "text_fr", "audio"]


def parse_modalities(modalities_arg: Optional[str]) -> list[str]:
    if not modalities_arg:
        return ["text_en", "text_es", "audio"]

    modalities = [item.strip() for item in modalities_arg.split(",") if item.strip()]
    invalid_modalities = [item for item in modalities if item not in SUPPORTED_MODALITIES]
    if invalid_modalities:
        raise ValueError(
            f"Modalitati invalide: {invalid_modalities}. Alege dintre {SUPPORTED_MODALITIES}"
        )
    if "audio" not in modalities:
        raise ValueError("Profesorul CCMT pentru distillation trebuie sa includa modalitatea 'audio'.")
    return modalities


def load_teacher_training_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Teacher training config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file_handle:
        config = json.load(file_handle)
    if "model_architecture" in config:
        return config["model_architecture"]
    return config


def collect_teacher_inputs(batch: dict, modalities: list[str], device: str) -> dict[str, torch.Tensor]:
    return {
        f"{modality}_emb": batch[f"{modality}_emb"].to(device, non_blocking=True)
        for modality in modalities
    }


class AudioDistillationCollator:
    def __init__(self, teacher_modalities: list[str]):
        self.audio_collator = AudioCollator()
        self.teacher_modalities = teacher_modalities

    def __call__(self, batch: list[dict]) -> dict:
        audio_batch = self.audio_collator(batch)
        merged = dict(audio_batch)
        merged["file_ids"] = [sample.get("file_id") for sample in batch]
        for modality in self.teacher_modalities:
            merged[f"{modality}_emb"] = torch.stack([sample[f"{modality}_emb"] for sample in batch])
        return merged


class AudioDistillationTrainer:
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        num_labels: int = 3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
        self.label2id = {value: key for key, value in self.id2label.items()}
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def setup_student_with_lora(self, lora_r: int = 8, lora_alpha: int = 16):
        model = AutoModelForAudioClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
        )
        model.config.use_cache = False

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

    def load_teacher(self, checkpoint_dir: Path, modalities: list[str], embeddings_dir: Path):
        teacher_config = load_teacher_training_config(checkpoint_dir)
        sample_dataset = PrecomputedEmbeddingsDataset(
            embeddings_dir=str(embeddings_dir),
            partition="train",
            device="cpu",
            modalities=modalities,
        )
        embedding_dims = sample_dataset.get_embedding_dims()
        model = load_ccmt_only_model(
            device=self.device,
            text_en_dim=embedding_dims.get("text_en", teacher_config.get("text_en_dim", 768)),
            text_es_dim=embedding_dims.get("text_es", teacher_config.get("text_es_dim", 768)),
            text_de_dim=embedding_dims.get("text_de", teacher_config.get("text_de_dim", 768)),
            text_fr_dim=embedding_dims.get("text_fr", teacher_config.get("text_fr_dim", 768)),
            audio_dim=embedding_dims.get("audio", teacher_config.get("audio_dim", 768)),
            num_classes=teacher_config.get("num_classes", 3),
            ccmt_dim=teacher_config.get("ccmt_dim", 768),
            num_patches_per_modality=teacher_config.get("num_patches_per_modality", 100),
            ccmt_depth=teacher_config.get("ccmt_depth", 4),
            ccmt_heads=teacher_config.get("ccmt_heads", 4),
            ccmt_mlp_dim=teacher_config.get("ccmt_mlp_dim", 1024),
            ccmt_dropout=teacher_config.get("ccmt_dropout", 0.1),
            modalities=modalities,
        )

        model_path = checkpoint_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def compute_kd_loss(
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        eps = 1e-8
        teacher_probs = teacher_probs.clamp_min(eps)
        softened_teacher = torch.softmax(torch.log(teacher_probs) / temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
        return torch.nn.functional.kl_div(
            student_log_probs,
            softened_teacher,
            reduction="batchmean",
        ) * (temperature ** 2)

    def train_epoch(
        self,
        student_model,
        teacher_model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        use_amp,
        teacher_modalities: list[str],
        alpha: float,
        temperature: float,
        gradient_accumulation_steps: int = 1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> dict:
        student_model.train()
        teacher_model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        valid_steps = 0
        all_predictions = []
        all_labels = []

        ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        progress_bar = tqdm(train_loader, desc="Training KD", dynamic_ncols=True)

        for step, batch in enumerate(progress_bar):
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            teacher_inputs = collect_teacher_inputs(batch, teacher_modalities, self.device)

            autocast_ctx = torch.amp.autocast(dtype=amp_dtype, device_type="cuda") if use_amp else nullcontext()
            with autocast_ctx:
                student_outputs = student_model(input_values=input_values, attention_mask=attention_mask)
                student_logits = student_outputs.logits.float().view(-1, self.num_labels)

                with torch.no_grad():
                    teacher_probs = teacher_model(**teacher_inputs).float()

                ce_loss = ce_loss_fn(student_logits, labels.view(-1))
                kd_loss = self.compute_kd_loss(student_logits, teacher_probs, temperature)
                loss = ((1.0 - alpha) * ce_loss + alpha * kd_loss) / gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            predictions = student_logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

            total_loss += loss.item() * gradient_accumulation_steps
            total_ce_loss += ce_loss.item()
            total_kd_loss += kd_loss.item()
            valid_steps += 1
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss / valid_steps:.4f}",
                    "ce": f"{total_ce_loss / valid_steps:.4f}",
                    "kd": f"{total_kd_loss / valid_steps:.4f}",
                }
            )

        return {
            "loss": total_loss / valid_steps,
            "ce_loss": total_ce_loss / valid_steps,
            "kd_loss": total_kd_loss / valid_steps,
            "accuracy": float(np.mean(np.array(all_predictions) == np.array(all_labels))),
            "f1_macro": float(f1_score(all_labels, all_predictions, average="macro", zero_division=0)),
        }

    @torch.no_grad()
    def evaluate(self, student_model, val_loader, use_amp, class_weights: Optional[torch.Tensor] = None) -> tuple:
        student_model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            autocast_ctx = torch.amp.autocast(dtype=amp_dtype, device_type="cuda") if use_amp else nullcontext()
            with autocast_ctx:
                outputs = student_model(input_values=input_values, attention_mask=attention_mask)
                logits = outputs.logits.float().view(-1, self.num_labels)
                labels_flat = labels.view(-1)
                loss = loss_fn(logits, labels_flat)

            batch_size = labels_flat.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions = logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

        avg_loss = total_loss / total_samples
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
        return avg_loss, accuracy, f1_macro

    def _plot_training_curves(self, history: dict, checkpoint_dir: Path):
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_loss"], label="Training Loss", linewidth=2, marker="o")
        plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2, marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Knowledge Distillation Training Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(checkpoint_dir / "training_curves.pdf", format="pdf", bbox_inches="tight")
        plt.close()

    def train(
        self,
        train_dataset,
        val_dataset,
        checkpoint_dir: Path,
        teacher_checkpoint_dir: Path,
        teacher_embeddings_dir: Path,
        teacher_modalities: list[str],
        num_epochs=5,
        batch_size=8,
        learning_rate=3e-4,
        lora_r=8,
        lora_alpha=16,
        gradient_accumulation_steps=1,
        use_amp=True,
        num_workers=0,
        alpha: float = 0.5,
        temperature: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        start_time = time.time()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        student_model = self.setup_student_with_lora(lora_r=lora_r, lora_alpha=lora_alpha)
        teacher_model = self.load_teacher(teacher_checkpoint_dir, teacher_modalities, teacher_embeddings_dir)

        if self.device != "cuda":
            use_amp = False

        effective_num_workers = 0 if os.name == "nt" else max(0, num_workers)
        collator = AudioDistillationCollator(teacher_modalities)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=effective_num_workers,
            pin_memory=self.device == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=effective_num_workers,
            pin_memory=self.device == "cuda",
        )

        trainable_params = [param for param in student_model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        best_val_f1 = 0.0
        history = {
            "train_loss": [],
            "train_ce_loss": [],
            "train_kd_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        print("=" * 80)
        print("Audio Knowledge Distillation Training - WavLM Student / CCMT Teacher")
        print("=" * 80)
        print(f"Teacher checkpoint: {teacher_checkpoint_dir}")
        print(f"Teacher embeddings: {teacher_embeddings_dir}")
        print(f"Teacher modalities: {teacher_modalities}")
        print(f"Batch size: {batch_size} (Effective: {batch_size * gradient_accumulation_steps})")
        print(f"Alpha: {alpha}")
        print(f"Temperature: {temperature}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            train_metrics = self.train_epoch(
                student_model=student_model,
                teacher_model=teacher_model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=use_amp,
                teacher_modalities=teacher_modalities,
                alpha=alpha,
                temperature=temperature,
                gradient_accumulation_steps=gradient_accumulation_steps,
                class_weights=class_weights,
            )
            val_loss, val_acc, val_f1 = self.evaluate(student_model, val_loader, use_amp, class_weights)

            history["train_loss"].append(float(train_metrics["loss"]))
            history["train_ce_loss"].append(float(train_metrics["ce_loss"]))
            history["train_kd_loss"].append(float(train_metrics["kd_loss"]))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            history["val_f1"].append(float(val_f1))

            print(
                f"Train Loss: {train_metrics['loss']:.4f} | CE: {train_metrics['ce_loss']:.4f} | "
                f"KD: {train_metrics['kd_loss']:.4f} | Train F1: {train_metrics['f1_macro']:.4f}"
            )
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_path = checkpoint_dir / "best_model"
                print(f"New best distilled student! F1 Macro: {val_f1:.4f} - Saving...")
                try:
                    merged_model = student_model.merge_and_unload() if hasattr(student_model, "merge_and_unload") else student_model
                    merged_model.save_pretrained(best_model_path)
                except Exception as exc:
                    print(f"[WARN] Could not merge student LoRA adapters ({exc}). Saving adapter-only checkpoint.")
                    student_model.save_pretrained(best_model_path)
                self.feature_extractor.save_pretrained(best_model_path)

        elapsed_seconds = time.time() - start_time
        training_results = {
            "total_training_time": {
                "seconds": elapsed_seconds,
                "minutes": elapsed_seconds / 60.0,
                "hours": elapsed_seconds / 3600.0,
            },
            "best_model_metrics": {
                "val_f1_macro": float(best_val_f1),
            },
            "distillation_hyperparameters": {
                "alpha": alpha,
                "temperature": temperature,
                "teacher_checkpoint_dir": str(teacher_checkpoint_dir),
                "teacher_embeddings_dir": str(teacher_embeddings_dir),
                "teacher_modalities": teacher_modalities,
            },
            "student_hyperparameters": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            "training_curves": history,
        }
        with (checkpoint_dir / "training_results.json").open("w", encoding="utf-8") as file_handle:
            json.dump(training_results, file_handle, indent=2)
        self._plot_training_curves(history, checkpoint_dir)
        print(f"\nTraining complete. Best Validation F1 Macro: {best_val_f1:.4f}")
        print(f"Model saved to: {checkpoint_dir / 'best_model'}")


def build_audio_teacher_dataset(
    partition: str,
    feature_extractor,
    embeddings_dir: Path,
    teacher_modalities: list[str],
) -> AudioTeacherDistillationDataset:
    data_dir = PROJECT_ROOT / "MSP_Podcast"
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    partition_map = {"train": "Train", "val": "Development", "test1": "Test1"}
    msp_dataset = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition=partition_map[partition],
        modalities=["audio"],
    )
    audio_dataset = AudioWaveLMDataset(
        msp_dataset,
        feature_extractor,
        max_seconds=5,
        do_resample=False,
        label_key="label_id",
        include_attention_mask=True,
    )
    teacher_dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition=partition,
        device="cpu",
        modalities=teacher_modalities,
    )
    return AudioTeacherDistillationDataset(audio_dataset, teacher_dataset)


def main():
    parser = argparse.ArgumentParser(description="Train a WavLM student via knowledge distillation from a CCMT teacher")
    parser.add_argument("--teacher-checkpoint-dir", type=str, required=True, help="Checkpoint directory containing the trained CCMT teacher")
    parser.add_argument("--teacher-embeddings-dir", type=str, required=True, help="Embeddings directory used by the CCMT teacher")
    parser.add_argument("--teacher-modalities", type=str, default="text_en,text_es,audio", help="Comma-separated modalities used by the teacher")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/wavlm_audio_kd", help="Output checkpoint directory for the distilled student")
    parser.add_argument("--alpha", type=float, default=None, help="Weight for the KD loss term")
    parser.add_argument("--temperature", type=float, default=None, help="Distillation temperature")
    args = parser.parse_args()

    train_config = get_training_config("wavlm_audio_kd", PROJECT_ROOT / "configs" / "training_config.json")
    teacher_modalities = parse_modalities(args.teacher_modalities)
    alpha = float(args.alpha if args.alpha is not None else train_config["alpha"])
    temperature = float(args.temperature if args.temperature is not None else train_config["temperature"])

    trainer = AudioDistillationTrainer()
    train_dataset = build_audio_teacher_dataset(
        partition="train",
        feature_extractor=trainer.feature_extractor,
        embeddings_dir=Path(args.teacher_embeddings_dir),
        teacher_modalities=teacher_modalities,
    )
    val_dataset = build_audio_teacher_dataset(
        partition="val",
        feature_extractor=trainer.feature_extractor,
        embeddings_dir=Path(args.teacher_embeddings_dir),
        teacher_modalities=teacher_modalities,
    )

    data_dir = PROJECT_ROOT / "MSP_Podcast"
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    class_weights_dataset = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition="Train",
        modalities=[],
    )
    all_train_labels = class_weights_dataset.metadata["label_id"].astype(int).to_numpy()
    raw_weights = class_weights_dataset.get_class_weights(all_train_labels, device=trainer.device)
    class_weights = torch.sqrt(raw_weights)

    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=Path(args.checkpoint_dir),
        teacher_checkpoint_dir=Path(args.teacher_checkpoint_dir),
        teacher_embeddings_dir=Path(args.teacher_embeddings_dir),
        teacher_modalities=teacher_modalities,
        num_epochs=int(train_config["num_epochs"]),
        batch_size=int(train_config["batch_size"]),
        learning_rate=float(train_config["learning_rate"]),
        lora_r=int(train_config["lora_r"]),
        lora_alpha=int(train_config["lora_alpha"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        use_amp=bool(train_config["use_amp"]),
        num_workers=int(train_config.get("num_workers", 0)),
        alpha=alpha,
        temperature=temperature,
        class_weights=class_weights,
    )


if __name__ == "__main__":
    main()