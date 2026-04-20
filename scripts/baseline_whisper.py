#!/usr/bin/env python3
"""
Whisper Audio-Only Baseline — MSP-Podcast 3-Class Sentiment
Input : audio (.wav) only
Output: unsatisfied (0) / neutral (1) / satisfied (2)

Usage:
    python scripts/baseline_whisper.py
"""

from __future__ import annotations

import json
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    WhisperFeatureExtractor,
    WhisperForAudioClassification,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # paths (relative to project root)
    data_dir:        str = "MSP_Podcast"
    checkpoint_dir:  str = "checkpoints/whisper_baseline"

    # model
    model_name:  str = "openai/whisper-base"
    num_labels:  int = 3

    # training
    num_epochs:                  int   = 10
    batch_size:                  int   = 8
    gradient_accumulation_steps: int   = 4
    learning_rate:               float = 1e-4
    warmup_ratio:                float = 0.1
    weight_decay:                float = 0.01
    max_grad_norm:               float = 1.0

    # audio
    target_sample_rate: int = 16_000

    # hardware
    use_amp:     bool = True
    num_workers: int  = 0   # keep 0 on Windows to avoid multiprocessing issues


# ─────────────────────────────────────────────────────────────────────────────
# Label definitions  (same mapping as src/data/dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

ID2LABEL: Dict[int, str] = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}


def _map_emotions(df: pd.DataFrame) -> pd.Series:
    """
    Collapse MSP-Podcast emotion codes into 3 sentiment classes.
    Mirrors the vectorised mapping in src/data/dataset.py.
    """
    emotion = df["EmoClass"]
    valence = df["EmoVal"].astype(float)

    is_negative = emotion.isin(["Ang", "Sad", "Dis", "Con", "Fea"])
    is_happy    = emotion == "Hap"
    is_neutral  = emotion == "Neu"
    is_low_val  = valence <= 3.5
    is_high_val = valence >= 4.5

    result = pd.Series("neutral", index=df.index, dtype=str)
    result[is_negative] = "unsatisfied"
    result[is_happy]    = "satisfied"
    result[is_neutral]  = "neutral"

    # Surprise / Other / No-agreement — fall back to valence score
    mask_other = ~(is_negative | is_happy | is_neutral)
    result[mask_other & is_low_val]  = "unsatisfied"
    result[mask_other & is_high_val] = "satisfied"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MSPWhisperDataset(Dataset):
    """
    Loads raw waveforms and 3-class sentiment labels from MSP-Podcast.
    Audio is resampled to 16 kHz and converted to mono on __getitem__.
    """

    def __init__(
        self,
        audio_root: str,
        labels_csv: str,
        partition: str,
        target_sr: int = 16_000,
    ) -> None:
        self.audio_root = audio_root
        self.target_sr  = target_sr
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        df = pd.read_csv(labels_csv)
        df = df[df["Split_Set"] == partition].reset_index(drop=True)
        df["sentiment"] = _map_emotions(df)
        df = df.dropna(subset=["sentiment"])
        df["label"] = df["sentiment"].map(LABEL2ID).astype("Int64")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        self.meta = df[["FileName", "label"]].reset_index(drop=True)

        counts = df["sentiment"].value_counts().reindex(
            ["unsatisfied", "neutral", "satisfied"], fill_value=0
        )
        print(f"  [{partition:12s}] {len(self.meta):>7,} samples — "
              f"unsatisfied: {counts['unsatisfied']:>6,}  "
              f"neutral: {counts['neutral']:>6,}  "
              f"satisfied: {counts['satisfied']:>6,}")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row   = self.meta.iloc[idx]
        fname = row["FileName"]

        audio_path = os.path.join(self.audio_root, fname)
        if not audio_path.endswith(".wav"):
            audio_path += ".wav"

        try:
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            waveform  = torch.from_numpy(audio).T          # [C, T]
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = self._resample(waveform, sr).squeeze(0)  # [T]
        except Exception as exc:
            print(f"[WARN] Could not load {audio_path}: {exc}")
            waveform = torch.zeros(self.target_sr, dtype=torch.float32)

        return {"waveform": waveform, "label": int(row["label"])}

    # ------------------------------------------------------------------

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return waveform
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr
            )
        return self._resamplers[sr](waveform)

    def get_class_weights(self, device: str) -> torch.Tensor:
        labels  = self.meta["label"].values
        classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        return torch.tensor(weights, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class WhisperCollator:
    """
    Converts a list of raw waveforms to Whisper log-mel spectrograms.
    WhisperFeatureExtractor pads / truncates to exactly 30 s (3000 frames).
    """

    def __init__(self, feature_extractor: WhisperFeatureExtractor) -> None:
        self.fe = feature_extractor

    def __call__(self, batch: list) -> dict:
        waveforms = [item["waveform"].numpy() for item in batch]
        labels    = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        inputs = self.fe(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {"input_features": inputs.input_features, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Config, device: str) -> WhisperForAudioClassification:
    model = WhisperForAudioClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Freeze the CNN front-end (conv1, conv2, embed_positions) —
    # these low-level feature extractors are stable and do not need
    # fine-tuning for downstream classification.
    frozen = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ("encoder.conv", "encoder.embed_positions")):
            param.requires_grad = False
            frozen += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Params — trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)  frozen: {frozen:,}")

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: WhisperForAudioClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    cfg: Config,
    loss_fn: torch.nn.Module,
) -> float:
    model.train()
    total_loss, steps = 0.0, 0
    use_amp  = cfg.use_amp and device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        feats  = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        ctx = torch.amp.autocast(dtype=amp_dtype, device_type="cuda") if use_amp else nullcontext()
        with ctx:
            logits = model(input_features=feats).logits.float()
            loss   = loss_fn(logits, labels) / cfg.gradient_accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        actual = loss.item() * cfg.gradient_accumulation_steps
        total_loss += actual
        steps += 1
        pbar.set_postfix({"loss": f"{actual:.4f}"})

    return total_loss / steps if steps else float("nan")


@torch.no_grad()
def evaluate(
    model: WhisperForAudioClassification,
    loader: DataLoader,
    device: str,
    cfg: Config,
    loss_fn: torch.nn.Module,
) -> tuple:
    model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []
    use_amp   = cfg.use_amp and device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for batch in tqdm(loader, desc="  Eval ", leave=False, dynamic_ncols=True):
        feats  = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        ctx = torch.amp.autocast(dtype=amp_dtype, device_type="cuda") if use_amp else nullcontext()
        with ctx:
            logits = model(input_features=feats).logits.float()
            loss   = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_n    += labels.size(0)
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_n
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, float(accuracy), float(f1_macro), all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curves(train_losses: list, val_losses: list, out_path: Path) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses,   "r-s", label="Val Loss",   linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Whisper Baseline — Training Curves")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


def save_confusion_matrix(
    labels: list, preds: list, num_labels: int, out_path: Path, title: str = "Confusion Matrix"
) -> None:
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    tick_labels = [ID2LABEL[i] for i in range(num_labels)]
    ax.set_xticks(range(num_labels)); ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(num_labels)); ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(num_labels):
        for j in range(num_labels):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = Path(cfg.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    data_dir   = Path(cfg.data_dir)
    labels_csv = str(data_dir / "Labels" / "Labels" / "labels_consensus.csv")
    audio_root = str(data_dir / "Audios" / "Audios")

    print("=" * 70)
    print("Whisper Audio-Only Baseline — MSP-Podcast 3-Class Sentiment")
    print("=" * 70)
    print(f"Model : {cfg.model_name}")
    print(f"Device: {device}")
    print(f"Output: {ckpt}\n")

    # ── Datasets ────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = MSPWhisperDataset(audio_root, labels_csv, "Train",       cfg.target_sample_rate)
    val_ds   = MSPWhisperDataset(audio_root, labels_csv, "Development", cfg.target_sample_rate)
    test_ds  = MSPWhisperDataset(audio_root, labels_csv, "Test1",       cfg.target_sample_rate)

    fe       = WhisperFeatureExtractor.from_pretrained(cfg.model_name)
    collator = WhisperCollator(fe)
    pin_mem  = device == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collator, num_workers=cfg.num_workers, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=cfg.num_workers, pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=cfg.num_workers, pin_memory=pin_mem,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(cfg, device)

    class_weights = train_ds.get_class_weights(device)
    loss_fn       = torch.nn.CrossEntropyLoss(weight=class_weights)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))

    # ── Training loop ────────────────────────────────────────────────────────
    best_f1    = 0.0
    best_path  = ckpt / "best_model"
    train_losses, val_losses = [], []
    t0 = time.time()

    print(f"\nStarting training — {cfg.num_epochs} epochs")
    print(f"  Effective batch size : {cfg.batch_size * cfg.gradient_accumulation_steps}")
    print(f"  Learning rate        : {cfg.learning_rate}")
    print(f"  Class weights        : {class_weights.cpu().numpy().round(3)}\n")

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        print("  " + "-" * 50)

        tr_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, cfg, loss_fn
        )
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, device, cfg, loss_fn)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"  Train loss : {tr_loss:.4f}")
        print(f"  Val   loss : {vl_loss:.4f}  |  acc: {vl_acc:.4f}  |  F1-macro: {vl_f1:.4f}")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            model.save_pretrained(best_path)
            fe.save_pretrained(best_path)
            print(f"  → Best model saved  (F1-macro = {vl_f1:.4f})")

        print()

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed / 60:.1f} min   Best val F1-macro: {best_f1:.4f}")

    # ── Persist training artefacts ───────────────────────────────────────────
    save_training_curves(train_losses, val_losses, ckpt / "training_curves.pdf")

    with open(ckpt / "training_results.json", "w") as f:
        json.dump(
            {
                "model": cfg.model_name,
                "best_val_f1_macro": best_f1,
                "training_time_minutes": elapsed / 60,
                "hyperparameters": asdict(cfg),
                "curves": {
                    "train_losses": [float(l) for l in train_losses],
                    "val_losses":   [float(l) for l in val_losses],
                },
            },
            f,
            indent=2,
        )

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Test Evaluation (Test1)")
    print("=" * 70)
    print(f"Loading best checkpoint from {best_path} ...")

    best_model = WhisperForAudioClassification.from_pretrained(best_path).to(device)

    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        best_model, test_loader, device, cfg, loss_fn
    )

    report = classification_report(
        test_labels, test_preds,
        target_names=[ID2LABEL[i] for i in range(cfg.num_labels)],
        digits=4,
    )
    print(f"\n  Accuracy : {test_acc:.4f}")
    print(f"  F1-macro : {test_f1:.4f}\n")
    print(report)

    save_confusion_matrix(
        test_labels, test_preds, cfg.num_labels,
        ckpt / "test1_confusion_matrix.pdf",
        title="Whisper Baseline — Test1 Confusion Matrix",
    )

    with open(ckpt / "test_results.json", "w") as f:
        json.dump(
            {
                "partition": "Test1",
                "accuracy":  float(test_acc),
                "f1_macro":  float(test_f1),
                "classification_report": classification_report(
                    test_labels, test_preds,
                    target_names=[ID2LABEL[i] for i in range(cfg.num_labels)],
                    output_dict=True,
                ),
            },
            f,
            indent=2,
        )

    print(f"All results saved to {ckpt}/")


if __name__ == "__main__":
    main()
