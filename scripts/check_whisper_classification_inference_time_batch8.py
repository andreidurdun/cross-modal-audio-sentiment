from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoConfig
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import soundfile as sf
import torchaudio


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_handle:
        import json
        payload = json.load(file_handle)
    if isinstance(payload, dict):
        return payload
    return {}


def _map_emotions(df: pd.DataFrame) -> pd.Series:
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

    mask_other = ~(is_negative | is_happy | is_neutral)
    result[mask_other & is_low_val]  = "unsatisfied"
    result[mask_other & is_high_val] = "satisfied"

    return result


class MSPWhisperDataset(Dataset):
    def __init__(
        self,
        audio_root: str,
        labels_csv: str,
        partition: str,
        target_sr: int = 16_000,
    ) -> None:
        self.audio_root = audio_root
        self.target_sr  = target_sr
        self._resamplers = {}

        df = pd.read_csv(labels_csv)
        df = df[df["Split_Set"] == partition].reset_index(drop=True)
        df["sentiment"] = _map_emotions(df)
        df = df.dropna(subset=["sentiment"])
        df["label"] = df["sentiment"].map({"unsatisfied": 0, "neutral": 1, "satisfied": 2}).astype("Int64")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        self.meta = df[["FileName", "label"]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row   = self.meta.iloc[idx]
        fname = row["FileName"]

        audio_path = Path(self.audio_root) / fname
        if not audio_path.suffix:
            audio_path = audio_path.with_suffix(".wav")

        try:
            audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            waveform  = torch.from_numpy(audio).T
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = self._resample(waveform, sr).squeeze(0)
        except Exception as exc:
            print(f"[WARN] Could not load {audio_path}: {exc}")
            waveform = torch.zeros(self.target_sr, dtype=torch.float32)

        return {"waveform": waveform, "label": int(row["label"])}

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return waveform
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr
            )
        return self._resamplers[sr](waveform)


class WhisperCollator:
    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, batch):
        waveforms = [item["waveform"].numpy() for item in batch]
        labels = [item["label"] for item in batch]

        input_features = []
        for waveform in waveforms:
            inputs = self.fe(waveform, sampling_rate=16000, return_tensors="pt")
            input_features.append(inputs["input_features"].squeeze(0))

        input_features = torch.stack(input_features)

        return {
            "input_features": input_features,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_handle:
        import json
        payload = json.load(file_handle)
    if isinstance(payload, dict):
        return payload
    return {}


def move_batch_to_device(batch: Any, device: str) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if hasattr(batch, "to") and callable(getattr(batch, "to")):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(item, device) for item in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(item, device) for item in batch)
    return batch


def benchmark_forward(
    model,
    batch: Any,
    forward_fn: Callable[[Any], Any],
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    model.eval()
    batch_on_device = move_batch_to_device(batch, device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = forward_fn(batch_on_device)
        if device == "cuda":
            torch.cuda.synchronize()

        elapsed_times_ms: list[float] = []
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = forward_fn(batch_on_device)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_times_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = sum(elapsed_times_ms) / len(elapsed_times_ms)
    min_ms = min(elapsed_times_ms)
    max_ms = max(elapsed_times_ms)

    effective_batch_size = batch["input_features"].shape[0]
    throughput = (effective_batch_size * 1000.0 / mean_ms) if mean_ms > 0 and effective_batch_size > 0 else 0.0

    return {
        "batch_size": float(effective_batch_size),
        "mean_inference_ms": float(mean_ms),
        "min_inference_ms": float(min_ms),
        "max_inference_ms": float(max_ms),
        "throughput_samples_per_sec": float(throughput),
    }


def main() -> None:
    batch_size = 8
    warmup = 3
    repeats = 20
    partition = "Development"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = PROJECT_ROOT / "results" / "whisper_baseline"

    data_dir = PROJECT_ROOT / "MSP_Podcast"
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    audio_root = str(data_dir / "Audios")

    # Load dataset
    dataset = MSPWhisperDataset(audio_root, str(labels_csv), partition)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    collator = WhisperCollator(feature_extractor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Load model
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained("openai/whisper-base", config=config).to(device)

    def forward_fn(batch: dict[str, torch.Tensor]):
        return model(input_features=batch["input_features"])

    # Get a batch
    batch = next(iter(loader))

    # Benchmark
    metrics = benchmark_forward(model, batch, forward_fn, device, warmup, repeats)

    print("=" * 60)
    print("Whisper Classification Model Benchmark Results:")
    print(f"Batch size: {metrics['batch_size']}")
    print(f"Mean inference time: {metrics['mean_inference_ms']:.2f} ms")
    print(f"Min inference time: {metrics['min_inference_ms']:.2f} ms")
    print(f"Max inference time: {metrics['max_inference_ms']:.2f} ms")
    print(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/s")
    print("=" * 60)

    # Save results to JSON
    import json
    output_data = {
        "model": "whisper_classification",
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "device": device,
        "metrics": metrics
    }
    output_path = PROJECT_ROOT / "results" / "whisper_classification_inference_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()