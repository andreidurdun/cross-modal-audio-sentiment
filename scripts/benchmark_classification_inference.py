from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from peft import AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoTokenizer, DataCollatorWithPadding

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from src.data.audio_datasets import AudioCollator, AudioWaveLMDataset
from src.data.dataset import MSP_Podcast_Dataset
from src.data.text_datasets import TextEncoderDataset
from src.models import load_full_multimodal_model
from src.preprocessing.audio_processor import AudioProcessor, AudioProcessorConfig
from src.preprocessing.transcriber import AudioTranscriber, TranscriberConfig
from src.preprocessing.translator import NLLBTranslator, NLLBTranslatorDE, NLLBTranslatorFR
from src.utils.peft_audio import load_peft_audio_classification_model


SUPPORTED_TEXT_MODALITIES = ["text_en", "text_es", "text_de", "text_fr"]


def infer_modalities_from_checkpoint_name(checkpoint_name: str) -> list[str]:
    tokens = checkpoint_name.split("_")
    modalities: list[str] = []
    index = 0

    while index < len(tokens):
        if index + 1 < len(tokens):
            text_modality = f"{tokens[index]}_{tokens[index + 1]}"
            if text_modality in SUPPORTED_TEXT_MODALITIES:
                modalities.append(text_modality)
                index += 2
                continue

        if tokens[index] == "audio":
            modalities.append("audio")

        index += 1

    if modalities:
        return modalities
    return ["text_en", "text_es", "audio"]


def resolve_text_transcript_path(data_dir: Path, modality: str) -> Path:
    suffix = modality.split("_", maxsplit=1)[1]
    return data_dir / f"Transcription_{suffix}.json"


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if isinstance(payload, dict):
        return payload
    return {}


def should_include_checkpoint(child: Path) -> bool:
    name = child.name
    if not child.is_dir():
        return False
    if name == "regression_pipeline" or "regression" in name:
        return False
    if "retrained" in name and not name.startswith("wavlm_audio_kd"):
        return False
    return True


def infer_text_num_labels(checkpoint_dir: Path, default: int = 3) -> int:
    training_results = load_json_if_exists(checkpoint_dir / "training_results.json")
    explicit_num_labels = training_results.get("num_labels")
    if isinstance(explicit_num_labels, int):
        return explicit_num_labels

    adapter_weights = checkpoint_dir / "adapter_model.safetensors"
    if adapter_weights.exists():
        try:
            from safetensors import safe_open

            candidate_suffixes = (
                "classifier.modules_to_save.default.weight",
                "classifier.modules_to_save.default.out_proj.weight",
                "score.modules_to_save.default.weight",
                "base_model.model.classifier.weight",
                "base_model.model.classifier.out_proj.weight",
                "classifier.weight",
                "score.weight",
            )

            with safe_open(str(adapter_weights), framework="pt", device="cpu") as file_handle:
                for key in file_handle.keys():
                    if key.endswith(candidate_suffixes):
                        return int(file_handle.get_tensor(key).shape[0])
        except ImportError:
            pass

    return default


def resolve_ccmt_checkpoint_path(checkpoint_dir: Path) -> Optional[Path]:
    primary = checkpoint_dir / "best_model.pt"
    if primary.exists():
        return primary
    return None


def create_msp_dataset(data_dir: Path, labels_csv: Path, partition: str, modalities: list[str]) -> MSP_Podcast_Dataset:
    transcript_paths = {
        "transcripts_en_json": str(data_dir / "Transcription_en.json") if "text_en" in modalities else None,
        "transcripts_es_json": str(data_dir / "Transcription_es.json") if "text_es" in modalities else None,
        "transcripts_de_json": str(data_dir / "Transcription_de.json") if "text_de" in modalities else None,
        "transcripts_fr_json": str(data_dir / "Transcription_fr.json") if "text_fr" in modalities else None,
    }
    return MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_en_json=transcript_paths["transcripts_en_json"],
        transcripts_es_json=transcript_paths["transcripts_es_json"],
        transcripts_de_json=transcript_paths["transcripts_de_json"],
        transcripts_fr_json=transcript_paths["transcripts_fr_json"],
        partition=partition,
        modalities=modalities,
    )


def pad_audio_tensors(audio_tensors: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(int(tensor.shape[-1]) for tensor in audio_tensors)
    padded_audio = torch.zeros((len(audio_tensors), max_length), dtype=torch.float32)
    for index, tensor in enumerate(audio_tensors):
        padded_audio[index, : tensor.shape[-1]] = tensor.to(torch.float32)
    return padded_audio


def collate_audio_path_samples(audio_root: Path, batch: list[dict[str, Any]]) -> dict[str, Any]:
    audio_paths: list[str] = []
    for sample in batch:
        audio_path = audio_root / str(sample["file_id"])
        if audio_path.suffix.lower() != ".wav":
            audio_path = audio_path.with_suffix(".wav")
        audio_paths.append(str(audio_path))

    return {
        "labels": torch.tensor([sample["label_id"] for sample in batch], dtype=torch.long),
        "audio_paths": audio_paths,
    }


def create_translator_map(modalities: list[str]) -> dict[str, Any]:
    translators: dict[str, Any] = {}
    if "text_es" in modalities:
        translators["text_es"] = NLLBTranslator()
    if "text_de" in modalities:
        translators["text_de"] = NLLBTranslatorDE()
    if "text_fr" in modalities:
        translators["text_fr"] = NLLBTranslatorFR()
    return translators


def translate_batch_texts(translator: Any, texts: list[str], batch_size: int) -> list[str]:
    translated: list[str] = []
    for start_index in range(0, len(texts), batch_size):
        translated.extend(translator.translate_batch(texts[start_index : start_index + batch_size]))
    return translated


def collate_multimodal_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {
        "labels": torch.tensor([sample["label_id"] for sample in batch], dtype=torch.long),
    }

    text_modalities = [key for key in batch[0].keys() if key.startswith("text_")]
    for modality in text_modalities:
        collated[modality] = [str(sample.get(modality, "")) for sample in batch]

    if "audio" in batch[0]:
        audio_tensors = [sample["audio"] for sample in batch]
        max_length = max(int(tensor.shape[-1]) for tensor in audio_tensors)
        padded_audio = torch.zeros((len(audio_tensors), max_length), dtype=torch.float32)
        for index, tensor in enumerate(audio_tensors):
            padded_audio[index, : tensor.shape[-1]] = tensor.to(torch.float32)
        collated["audio"] = padded_audio

    return collated


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

    labels = batch_on_device.get("labels") if hasattr(batch_on_device, "get") else None
    effective_batch_size = int(labels.shape[0]) if isinstance(labels, torch.Tensor) else 0
    throughput = (effective_batch_size * 1000.0 / mean_ms) if mean_ms > 0 and effective_batch_size > 0 else 0.0

    return {
        "batch_size": float(effective_batch_size),
        "mean_inference_ms": float(mean_ms),
        "min_inference_ms": float(min_ms),
        "max_inference_ms": float(max_ms),
        "throughput_samples_per_sec": float(throughput),
    }


def build_text_benchmark(modality: str, checkpoint_dir: Path, partition: str, batch_size: int, device: str):
    data_dir = PROJECT_ROOT / "MSP_Podcast"
    transcript_path = resolve_text_transcript_path(data_dir, modality)
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    best_model_path = checkpoint_dir / "best_model"

    _ = transcript_path
    base_dataset = create_msp_dataset(data_dir, labels_csv, partition, [modality])
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    dataset = TextEncoderDataset(
        base_dataset,
        tokenizer,
        text_fields=[modality],
        max_length=512,
        padding=True,
        sanitize_input_ids=True,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        best_model_path,
        num_labels=infer_text_num_labels(best_model_path),
        ignore_mismatched_sizes=True,
    ).to(device)

    def forward_fn(batch: dict[str, torch.Tensor]):
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(batch["input_ids"], device=batch["input_ids"].device)
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=token_type_ids,
        )

    return loader, model, forward_fn


def build_audio_benchmark(checkpoint_dir: Path, partition: str, batch_size: int, device: str):
    data_dir = PROJECT_ROOT / "MSP_Podcast"
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    best_model_path = checkpoint_dir / "best_model"

    base_dataset = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition=partition,
        modalities=["audio"],
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(best_model_path)
    dataset = AudioWaveLMDataset(
        base_dataset,
        feature_extractor,
        max_seconds=None,
        do_resample=True,
        label_key="label_id",
        include_attention_mask=True,
        extractor_padding=True,
        extractor_truncation=True,
        extractor_max_length=160000,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=AudioCollator())
    model = load_peft_audio_classification_model(best_model_path, num_labels=3).to(device)

    def forward_fn(batch: dict[str, torch.Tensor]):
        return model(input_values=batch["input_values"], attention_mask=batch["attention_mask"])

    return loader, model, forward_fn


def build_ccmt_benchmark(checkpoint_dir: Path, partition: str, batch_size: int, device: str):
    data_dir = PROJECT_ROOT / "MSP_Podcast"
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    saved_config = load_json_if_exists(checkpoint_dir / "training_config.json")
    model_config = saved_config.get("model_architecture", {})
    modalities = list(model_config.get("modalities") or infer_modalities_from_checkpoint_name(checkpoint_dir.name))
    checkpoint_path = resolve_ccmt_checkpoint_path(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No root best_model.pt found in {checkpoint_dir}")
    dataset = create_msp_dataset(data_dir, labels_csv, partition, modalities)
    model = load_full_multimodal_model(
        device=device,
        text_en_checkpoint=PROJECT_ROOT / "checkpoints" / "roberta_text_en",
        text_es_checkpoint=PROJECT_ROOT / "checkpoints" / "roberta_text_es",
        text_de_checkpoint=PROJECT_ROOT / "checkpoints" / "roberta_text_de",
        text_fr_checkpoint=PROJECT_ROOT / "checkpoints" / "roberta_text_fr",
        audio_checkpoint=PROJECT_ROOT / "checkpoints" / "wavlm_audio",
        num_classes=int(model_config.get("num_classes", 3)),
        ccmt_dim=int(model_config.get("ccmt_dim", 768)),
        num_patches_per_modality=int(model_config.get("num_patches_per_modality", 100)),
        ccmt_depth=int(model_config.get("ccmt_depth", 4)),
        ccmt_heads=int(model_config.get("ccmt_heads", 4)),
        ccmt_mlp_dim=int(model_config.get("ccmt_mlp_dim", 1024)),
        freeze_backbones=True,
        projection_dim=None,
        modalities=modalities,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_samples)

    def forward_fn(batch: dict[str, Any]):
        inputs = {modality: batch[modality] for modality in modalities}
        return model(**inputs)

    return loader, model.to(device), forward_fn, modalities


def build_ccmt_end_to_end_benchmark(
    checkpoint_dir: Path,
    partition: str,
    batch_size: int,
    device: str,
    transcription_batch_size: int,
    translation_batch_size: int,
):
    data_dir = PROJECT_ROOT / "MSP_Podcast"
    audio_root = data_dir / "Audios"
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    saved_config = load_json_if_exists(checkpoint_dir / "training_config.json")
    model_config = saved_config.get("model_architecture", {})
    modalities = list(model_config.get("modalities") or infer_modalities_from_checkpoint_name(checkpoint_dir.name))
    checkpoint_path = resolve_ccmt_checkpoint_path(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No root best_model.pt found in {checkpoint_dir}")

    dataset = create_msp_dataset(data_dir, labels_csv, partition, [])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_audio_path_samples(audio_root, batch),
    )

    model = load_full_multimodal_model(
        device=device,
        text_en_checkpoint=PROJECT_ROOT / "checkpoints" / "roberta_text_en",
        text_es_checkpoint=PROJECT_ROOT / "checkpoints" / "roberta_text_es",
        text_de_checkpoint=PROJECT_ROOT / "checkpoints" / "roberta_text_de",
        text_fr_checkpoint=PROJECT_ROOT / "checkpoints" / "roberta_text_fr",
        audio_checkpoint=PROJECT_ROOT / "checkpoints" / "wavlm_audio",
        num_classes=int(model_config.get("num_classes", 3)),
        ccmt_dim=int(model_config.get("ccmt_dim", 768)),
        num_patches_per_modality=int(model_config.get("num_patches_per_modality", 100)),
        ccmt_depth=int(model_config.get("ccmt_depth", 4)),
        ccmt_heads=int(model_config.get("ccmt_heads", 4)),
        ccmt_mlp_dim=int(model_config.get("ccmt_mlp_dim", 1024)),
        freeze_backbones=True,
        projection_dim=None,
        modalities=modalities,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    audio_processor = AudioProcessor(AudioProcessorConfig(target_sample_rate=16000))
    transcriber = AudioTranscriber(TranscriberConfig(batch_size=transcription_batch_size))
    translators = create_translator_map(modalities)

    def forward_fn(batch: dict[str, Any]):
        audio_paths = list(batch["audio_paths"])
        text_en = transcriber.transcribe_batch(audio_paths, batch_size=transcription_batch_size)

        model_inputs: dict[str, Any] = {}
        if "text_en" in modalities:
            model_inputs["text_en"] = text_en
        if "text_es" in modalities:
            model_inputs["text_es"] = translate_batch_texts(translators["text_es"], text_en, translation_batch_size)
        if "text_de" in modalities:
            model_inputs["text_de"] = translate_batch_texts(translators["text_de"], text_en, translation_batch_size)
        if "text_fr" in modalities:
            model_inputs["text_fr"] = translate_batch_texts(translators["text_fr"], text_en, translation_batch_size)
        if "audio" in modalities:
            audio_tensors = [audio_processor.load_waveform(path) for path in audio_paths]
            model_inputs["audio"] = pad_audio_tensors(audio_tensors).to(device)

        return model(**model_inputs)

    return loader, model.to(device), forward_fn, modalities


def discover_classification_checkpoints(checkpoints_root: Path) -> list[Path]:
    discovered: list[Path] = []
    for child in sorted(checkpoints_root.iterdir()):
        if not should_include_checkpoint(child):
            continue
        if child.name.startswith("roberta_text_"):
            if not (child / "best_model" / "adapter_config.json").exists():
                continue
            discovered.append(child)
            continue
        if child.name.startswith("wavlm_audio"):
            if not (child / "best_model").exists():
                continue
            discovered.append(child)
            continue
        if child.name.startswith("ccmt_multimodal"):
            if not (child / "training_config.json").exists() or resolve_ccmt_checkpoint_path(child) is None:
                continue
            discovered.append(child)
    return discovered


def resolve_checkpoint_inputs(checkpoint_inputs: list[str], checkpoints_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for item in checkpoint_inputs:
        candidate = Path(item)
        if candidate.exists():
            resolved.append(candidate)
            continue

        root_candidate = checkpoints_root / item
        if root_candidate.exists():
            resolved.append(root_candidate)
            continue

        if not candidate.is_absolute():
            project_candidate = PROJECT_ROOT / item
            if project_candidate.exists():
                resolved.append(project_candidate)
                continue

        resolved.append(candidate)
    return resolved


def benchmark_model(
    checkpoint_dir: Path,
    partition: str,
    batch_size: int,
    warmup: int,
    repeats: int,
    device: str,
    ccmt_input_mode: str,
    transcription_batch_size: int,
    translation_batch_size: int,
) -> dict[str, Any]:
    name = checkpoint_dir.name
    print(f"\nBenchmarking {name}...")

    if name.startswith("roberta_text_"):
        modality = name.replace("roberta_", "", 1)
        loader, model, forward_fn = build_text_benchmark(modality, checkpoint_dir, partition, batch_size, device)
        batch = next(iter(loader))
        metrics = benchmark_forward(model, batch, forward_fn, device, warmup, repeats)
        metrics.update({"model_type": "text-classification", "modality": modality, "checkpoint_dir": str(checkpoint_dir)})
        return metrics

    if name.startswith("wavlm_audio"):
        loader, model, forward_fn = build_audio_benchmark(checkpoint_dir, partition, batch_size, device)
        batch = next(iter(loader))
        metrics = benchmark_forward(model, batch, forward_fn, device, warmup, repeats)
        metrics.update({"model_type": "audio-classification", "checkpoint_dir": str(checkpoint_dir)})
        return metrics

    if name.startswith("ccmt_multimodal"):
        if ccmt_input_mode == "wav-end-to-end":
            loader, model, forward_fn, modalities = build_ccmt_end_to_end_benchmark(
                checkpoint_dir,
                partition,
                batch_size,
                device,
                transcription_batch_size,
                translation_batch_size,
            )
        else:
            loader, model, forward_fn, modalities = build_ccmt_benchmark(checkpoint_dir, partition, batch_size, device)
        batch = next(iter(loader))
        metrics = benchmark_forward(model, batch, forward_fn, device, warmup, repeats)
        metrics.update(
            {
                "model_type": "ccmt-classification",
                "modalities": modalities,
                "includes_backbones": True,
                "ccmt_input_mode": ccmt_input_mode,
                "includes_transcription": ccmt_input_mode == "wav-end-to-end",
                "includes_translation": ccmt_input_mode == "wav-end-to-end" and any(
                    modality in modalities for modality in ["text_es", "text_de", "text_fr"]
                ),
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
        return metrics

    raise ValueError(f"Unsupported checkpoint family: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mini-batch inference time for classification models")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size used for benchmarking")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations before timing")
    parser.add_argument("--repeats", type=int, default=20, help="Number of timed forward passes")
    parser.add_argument(
        "--partition",
        type=str,
        default="Development",
        help="Dataset partition for raw-text/audio models. For CCMT this is mapped to val/test1 style names.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "classification_inference_benchmark.json",
        help="Path to the JSON file where benchmark results are saved",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of checkpoint directories to benchmark",
    )
    parser.add_argument(
        "--ccmt-input-mode",
        type=str,
        choices=["raw-modalities", "wav-end-to-end"],
        default="wav-end-to-end",
        help="How CCMT models are benchmarked: direct multimodal inputs or end-to-end from .wav with transcription and translation.",
    )
    parser.add_argument(
        "--transcription-batch-size",
        type=int,
        default=None,
        help="Batch size used by Faster-Whisper when CCMT runs in wav-end-to-end mode. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=None,
        help="Batch size used by NLLB translation when CCMT runs in wav-end-to-end mode. Defaults to --batch-size.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints_root = PROJECT_ROOT / "checkpoints"
    checkpoint_dirs = (
        resolve_checkpoint_inputs(args.checkpoints, checkpoints_root)
        if args.checkpoints
        else discover_classification_checkpoints(checkpoints_root)
    )
    transcription_batch_size = args.transcription_batch_size or args.batch_size
    translation_batch_size = args.translation_batch_size or args.batch_size

    existing_results = load_json_if_exists(args.output_json)
    existing_models = existing_results.get("models", {}) if isinstance(existing_results.get("models", {}), dict) else {}

    results: dict[str, Any] = {
        "device": device,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "ccmt_input_mode": args.ccmt_input_mode,
        "models": dict(existing_models),
    }

    for checkpoint_dir in checkpoint_dirs:
        try:
            metrics = benchmark_model(
                checkpoint_dir=checkpoint_dir,
                partition=args.partition,
                batch_size=args.batch_size,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
                ccmt_input_mode=args.ccmt_input_mode,
                transcription_batch_size=transcription_batch_size,
                translation_batch_size=translation_batch_size,
            )
            results["models"][checkpoint_dir.name] = metrics
            print(
                f"  mean={metrics['mean_inference_ms']:.2f} ms | "
                f"throughput={metrics['throughput_samples_per_sec']:.2f} samples/s"
            )
        except Exception as exc:  # noqa: BLE001
            results["models"][checkpoint_dir.name] = {
                "checkpoint_dir": str(checkpoint_dir),
                "error": str(exc),
            }
            print(f"  [WARN] {checkpoint_dir.name} failed: {exc}")
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2)

    print(f"\nSaved benchmark results to: {args.output_json}")


if __name__ == "__main__":
    main()