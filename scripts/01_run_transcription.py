"""Batch transcription entry point."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Dict, Sequence

import pandas as pd
from tqdm.auto import tqdm

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from src.preprocessing.transcriber import AudioTranscriber
from src.utils.helpers import set_seed


DEFAULT_DATASET_ROOT = Path("MSP_Podcast")
DEFAULT_PARTITIONS: Sequence[str] = ("Train", "Development")
DEFAULT_MAX_FILES: int | None = None
DEFAULT_SEED = 42
DEFAULT_OUTPUT_EN_DIR = Path("MSP_Podcast/Transcription_en")
DEFAULT_OUTPUT_JSON = Path("MSP_Podcast/Transcription_en.json")
DEFAULT_BATCH_SIZE = 8
DEFAULT_SAVE_EVERY = 50


def parse_partitions(partitions_arg: str | None) -> list[str]:
    if not partitions_arg:
        return list(DEFAULT_PARTITIONS)
    return [item.strip() for item in partitions_arg.split(",") if item.strip()]


def filter_partition_rows(df: pd.DataFrame, partition: str, max_files: int | None) -> pd.DataFrame:
    subset = df[df["Split_Set"] == partition].copy()
    if max_files is not None:
        subset = subset.head(max_files)
    if subset.empty:
        raise ValueError(f"Partition '{partition}' not found in metadata")
    return subset


def load_existing_transcripts(output_json: Path) -> Dict[str, str]:
    if not output_json.exists():
        return {}

    with output_json.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {output_json}, got {type(payload).__name__}")
    return {str(key): str(value) for key, value in payload.items()}


def save_transcripts(output_json: Path, transcripts: Dict[str, str]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file_handle:
        json.dump(transcripts, file_handle, ensure_ascii=False, indent=2)


def format_seconds(total_seconds: float | None) -> str:
    if total_seconds is None or not math.isfinite(total_seconds):
        return "--:--:--"

    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def process_partition(
    partition: str,
    rows: pd.DataFrame,
    audio_dir: Path,
    transcriber: AudioTranscriber,
    all_transcripts: Dict[str, str],
    output_json: Path,
    batch_size: int,
    save_every: int,
):
    saved_files = 0
    resumed_files = 0
    missing_files = 0
    transcripts: Dict[str, str] = {}
    pending_items: list[tuple[str, Path]] = []

    for file_id in rows["FileName"].tolist():
        audio_path = audio_dir / file_id
        if audio_path.suffix.lower() != ".wav":
            audio_path = audio_path.with_suffix(".wav")

        key = audio_path.stem
        if key in all_transcripts:
            resumed_files += 1
            continue

        if not audio_path.exists():
            print(f"[WARN] Missing audio for {file_id} -> {audio_path}")
            missing_files += 1
            continue

        pending_items.append((key, audio_path))

    if not pending_items:
        print(
            f"Partition {partition}: nimic nou de transcris "
            f"(resume={resumed_files}, missing={missing_files})"
        )
        return transcripts

    started_at = time.monotonic()
    progress_bar = tqdm(
        total=len(pending_items),
        desc=f"Transcribing {partition}",
        unit="file",
        dynamic_ncols=True,
    )

    for start_idx in range(0, len(pending_items), batch_size):
        batch_items = pending_items[start_idx : start_idx + batch_size]
        batch_keys = [key for key, _ in batch_items]
        batch_paths = [str(audio_path) for _, audio_path in batch_items]

        try:
            batch_texts = transcriber.transcribe_batch(batch_paths, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001 - log and continue
            tqdm.write(f"[WARN] Batch failed in {partition} ({batch_keys[0]} .. {batch_keys[-1]}): {exc}")
            batch_texts = []
            for key, audio_path in batch_items:
                try:
                    batch_texts.append(transcriber.transcribe(str(audio_path)))
                except Exception as sample_exc:  # noqa: BLE001 - log and continue
                    tqdm.write(f"[ERROR] Failed processing {audio_path.name}: {sample_exc}")
                    batch_texts.append("")

        for key, text_en in zip(batch_keys, batch_texts):
            if not text_en:
                continue
            transcripts[key] = text_en
            all_transcripts[key] = text_en
            saved_files += 1

        progress_bar.update(len(batch_items))

        processed_count = min(start_idx + len(batch_items), len(pending_items))
        elapsed = time.monotonic() - started_at
        rate = processed_count / elapsed if elapsed > 0 else 0.0
        remaining = len(pending_items) - processed_count
        eta_seconds = (remaining / rate) if rate > 0 else None
        progress_bar.set_postfix(
            rate=f"{rate:.2f}/s" if rate > 0 else "0.00/s",
            eta=format_seconds(eta_seconds),
            last=batch_keys[-1],
            saved=len(all_transcripts),
        )

        if saved_files and (saved_files % save_every == 0 or processed_count == len(pending_items)):
            save_transcripts(output_json, all_transcripts)
            tqdm.write(f"[SAVE] Saved {len(all_transcripts)} transcripts to {output_json}")

    progress_bar.close()
    print(
        f"Saved {saved_files} new transcripts for {partition} "
        f"(resume={resumed_files}, missing={missing_files}, elapsed={format_seconds(time.monotonic() - started_at)})"
    )
    return transcripts


def run_transcribe(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    partitions: Sequence[str] = DEFAULT_PARTITIONS,
    max_files: int | None = DEFAULT_MAX_FILES,
    seed: int = DEFAULT_SEED,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_every: int = DEFAULT_SAVE_EVERY,
):
    set_seed(seed)

    audio_dir = dataset_root / "Audios"
    metadata_csv = dataset_root / "Labels" / "labels_consensus.csv"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    transcriber = AudioTranscriber()

    all_transcripts = load_existing_transcripts(output_json)
    if all_transcripts:
        print(f"Loaded {len(all_transcripts)} existing transcripts from {output_json}")

    for partition in partitions:
        try:
            rows = filter_partition_rows(df, partition, max_files)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        partition_transcripts = process_partition(
            partition=partition,
            rows=rows,
            audio_dir=audio_dir,
            transcriber=transcriber,
            all_transcripts=all_transcripts,
            output_json=output_json,
            batch_size=batch_size,
            save_every=save_every,
        )
        all_transcripts.update(partition_transcripts)

    if all_transcripts:
        save_transcripts(output_json, all_transcripts)
        print(f"Saved {len(all_transcripts)} transcripts to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate English transcripts for selected MSP-Podcast partitions")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory. Default: MSP_Podcast",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default=",".join(DEFAULT_PARTITIONS),
        help="Comma-separated partitions to process. Example: Train,Development,Test1",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help="Optional limit for number of files per partition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path for generated transcripts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used by faster-whisper batched inference.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Save transcripts incrementally after this many new successful files.",
    )
    args = parser.parse_args()

    run_transcribe(
        dataset_root=args.dataset_root,
        partitions=parse_partitions(args.partitions),
        max_files=args.max_files,
        seed=args.seed,
        output_json=args.output_json,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )