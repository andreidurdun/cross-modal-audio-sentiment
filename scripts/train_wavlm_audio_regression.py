import argparse
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from src.data.audio_datasets import AudioWaveLMDataset
from src.data.dataset import MSP_Podcast_Dataset
from src.data.text_datasets import TextRegressionDataset
from src.utils import AudioRegressionTrainer, get_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train audio backbone for regression")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/wavlm_audio_regression",
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
        "wavlm_audio_regression",
        PROJECT_ROOT / "configs" / "training_config.json",
    )

    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    checkpoint_dir = Path(args.checkpoint_dir)

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    ensure_checkpoint_dir_available(checkpoint_dir, args.allow_overwrite)

    print("=" * 80)
    print("Loading MSP-Podcast Audio Data for Regression")
    print("=" * 80)

    train_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition="Train",
        modalities=["audio"],
    )
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition="Development",
        modalities=["audio"],
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    target_transform = lambda tensor: TextRegressionDataset.normalize_targets(tensor.to(torch.float32))

    train_dataset = AudioWaveLMDataset(
        train_dataset_msp,
        feature_extractor,
        max_seconds=5,
        do_resample=False,
        label_key="val_arousal",
        label_dtype=torch.float32,
        label_transform=target_transform,
        include_attention_mask=False,
    )
    val_dataset = AudioWaveLMDataset(
        val_dataset_msp,
        feature_extractor,
        max_seconds=5,
        do_resample=False,
        label_key="val_arousal",
        label_dtype=torch.float32,
        label_transform=target_transform,
        include_attention_mask=False,
    )

    trainer = AudioRegressionTrainer(model_name="microsoft/wavlm-base-plus")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        checkpoint_dir=checkpoint_dir,
        num_epochs=int(train_config["num_epochs"]),
        batch_size=int(train_config["batch_size"]),
        learning_rate=float(train_config["learning_rate"]),
        lora_r=int(train_config["lora_r"]),
        lora_alpha=int(train_config["lora_alpha"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        use_amp=bool(train_config["use_amp"]),
        num_workers=int(train_config.get("num_workers", 0)),
    )


if __name__ == "__main__":
    main()