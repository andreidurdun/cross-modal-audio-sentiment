import argparse
from pathlib import Path

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from src.data.dataset import MSP_Podcast_Dataset
from src.data.text_datasets import TextRegressionDataset
from src.utils import TextRegressionTrainer, get_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train German text backbone for regression")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/roberta_text_de_regression",
        help="Directorul in care se salveaza checkpoint-urile.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Permite suprascrierea unui director de checkpoint existent si ne-gol.",
    )
    return parser.parse_args()


def ensure_checkpoint_dir_available(checkpoint_dir: Path, allow_overwrite: bool) -> None:
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise FileExistsError(f"Checkpoint path exists and is not a directory: {checkpoint_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Checkpoint directory already exists and is not empty: {checkpoint_dir}. "
            "Use a different --checkpoint-dir or pass --allow-overwrite explicitly."
        )


def main():
    args = parse_args()
    train_config = get_training_config(
        "roberta_text_de_regression",
        PROJECT_ROOT / "configs" / "training_config.json",
    )

    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_de_json = data_dir / "Transcription_de.json"
    checkpoint_dir = Path(args.checkpoint_dir)

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_de_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_de_json}")
    ensure_checkpoint_dir_available(checkpoint_dir, args.allow_overwrite)

    print("=" * 80)
    print("Loading MSP-Podcast German Text Data for Regression")
    print("=" * 80)

    train_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_de_json=str(transcripts_de_json),
        partition="Train",
        modalities=["text_de"],
    )
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_de_json=str(transcripts_de_json),
        partition="Development",
        modalities=["text_de"],
    )

    trainer = TextRegressionTrainer(model_name="deepset/gbert-base")
    train_dataset = TextRegressionDataset(train_dataset_msp, trainer.tokenizer, text_fields=["text_de", "text_en"], max_length=128, padding="max_length")
    val_dataset = TextRegressionDataset(val_dataset_msp, trainer.tokenizer, text_fields=["text_de", "text_en"], max_length=128, padding="max_length")

    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        num_epochs=int(train_config["num_epochs"]),
        batch_size=int(train_config["batch_size"]),
        learning_rate=float(train_config["learning_rate"]),
        lora_r=int(train_config["lora_r"]),
        lora_alpha=int(train_config["lora_alpha"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
    )


if __name__ == "__main__":
    main()