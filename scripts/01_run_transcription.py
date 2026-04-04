"""Batch transcription entry point."""
from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Dict, Sequence
import pandas as pd

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


def filter_partition_rows(df: pd.DataFrame, partition: str, max_files: int | None) -> pd.DataFrame:
    subset = df[df["Split_Set"] == partition].copy()
    if max_files is not None:
        subset = subset.head(max_files)
    if subset.empty:
        raise ValueError(f"Partition '{partition}' not found in metadata")
    return subset


def process_partition(
    partition: str,
    rows: pd.DataFrame,
    audio_dir: Path,
    transcriber: AudioTranscriber,
):
    saved_files = 0
    transcripts: Dict[str, str] = {}

    for file_id in rows["FileName"].tolist():
        audio_path = audio_dir / file_id
        if audio_path.suffix.lower() != ".wav":
            audio_path = audio_path.with_suffix(".wav")

        if not audio_path.exists():
            print(f"[WARN] Missing audio for {file_id} -> {audio_path}")
            continue

        try:
            key = audio_path.stem
            text_en = transcriber.transcribe(str(audio_path))
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"[ERROR] Failed processing {file_id}: {exc}")
            continue

        transcripts[key] = text_en
        saved_files += 1
        print(f"[OK] {partition}: {key}")

    print(f"Saved {saved_files} transcripts for {partition}")
    return transcripts


def run_transcribe(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    partitions: Sequence[str] = DEFAULT_PARTITIONS,
    max_files: int | None = DEFAULT_MAX_FILES,
    seed: int = DEFAULT_SEED,
    output_json: Path = DEFAULT_OUTPUT_JSON,
):
    set_seed(seed)

    audio_dir = dataset_root / "Audios"
    metadata_csv = dataset_root / "Labels" / "labels_consensus.csv"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    transcriber = AudioTranscriber()


    all_transcripts: Dict[str, str] = {}

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
        )
        all_transcripts.update(partition_transcripts)

    if all_transcripts:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(all_transcripts, f, ensure_ascii=False)
        print(f"Saved {len(all_transcripts)} transcripts to {output_json}")


if __name__ == "__main__":
    run_transcribe()