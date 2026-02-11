"""Batch transcription entry point."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Sequence
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.preprocessing.transcriber import AudioTranscriber
from src.utils.helpers import set_seed


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_PARTITIONS: Sequence[str] = ("Train", "Validation")
DEFAULT_MAX_FILES: int | None = None
DEFAULT_SEED = 42
DEFAULT_OUTPUT_EN_DIR = Path("MSP_Podcast/Transcription_en")


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    output_en_dir: Path,
    transcriber: AudioTranscriber,
):
    output_en_dir.mkdir(parents=True, exist_ok=True)

    saved_files = 0

    for file_id in rows["FileName"].tolist():
        audio_path = audio_dir / file_id
        if audio_path.suffix.lower() != ".wav":
            audio_path = audio_path.with_suffix(".wav")

        if not audio_path.exists():
            print(f"[WARN] Missing audio for {file_id} -> {audio_path}")
            continue

        try:
            key = audio_path.stem
            transcript_path = output_en_dir / f"{key}.txt"
            if transcript_path.exists():
                continue

            text_en = transcriber.transcribe(str(audio_path))
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"[ERROR] Failed processing {file_id}: {exc}")
            continue

        transcript_path.write_text(text_en, encoding="utf-8")
        saved_files += 1
        print(f"[OK] {partition}: {key}")

    print(f"Saved {saved_files} transcripts to {output_en_dir}")


def run_transcribe(
    config_path: Path = DEFAULT_CONFIG_PATH,
    partitions: Sequence[str] = DEFAULT_PARTITIONS,
    max_files: int | None = DEFAULT_MAX_FILES,
    seed: int = DEFAULT_SEED,
    output_en_dir: Path = DEFAULT_OUTPUT_EN_DIR,
):
    set_seed(seed)

    cfg = load_config(config_path)
    dataset_root = Path(cfg["data"]["dataset_root"])
    audio_dir = dataset_root / "Audios"
    metadata_csv = dataset_root / "Labels" / "labels_consensus.csv"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    transcriber = AudioTranscriber()


    for partition in partitions:
        try:
            rows = filter_partition_rows(df, partition, max_files)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        process_partition(
            partition=partition,
            rows=rows,
            audio_dir=audio_dir,
            output_en_dir=output_en_dir,
            transcriber=transcriber,
        )


if __name__ == "__main__":
    run_transcribe()