from pathlib import Path

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from src.data.dataset import MSP_Podcast_Dataset
from src.data.text_datasets import TextRegressionDataset
from src.utils import TextRegressionTrainer, get_training_config


def main():
    train_config = get_training_config(
        "roberta_text_en_regression",
        PROJECT_ROOT / "configs" / "training_config.json",
    )

    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_en_json = data_dir / "Transcription_en.json"
    checkpoint_dir = Path("checkpoints/roberta_text_en_regression")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_en_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_en_json}")

    print("=" * 80)
    print("Loading MSP-Podcast English Text Data for Regression")
    print("=" * 80)

    train_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_en_json=str(transcripts_en_json),
        partition="Train",
        modalities=["text_en"],
    )
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_en_json=str(transcripts_en_json),
        partition="Development",
        modalities=["text_en"],
    )

    trainer = TextRegressionTrainer(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest")
    train_dataset = TextRegressionDataset(train_dataset_msp, trainer.tokenizer, text_fields=["text_en"], max_length=128, padding="max_length")
    val_dataset = TextRegressionDataset(val_dataset_msp, trainer.tokenizer, text_fields=["text_en"], max_length=128, padding="max_length")

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