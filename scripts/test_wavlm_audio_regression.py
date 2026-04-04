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
from src.utils import AudioRegressionTester


def main():
    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    audio_dir = data_dir / "Audios"
    checkpoint_dir = Path("checkpoints/wavlm_audio_regression")
    output_dir = Path("results/wavlm_audio_regression")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    print("=" * 80)
    print("Loading MSP-Podcast Audio Data for Regression Testing")
    print("=" * 80)

    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(audio_dir),
        labels_csv=str(labels_csv),
        partition="Development",
        modalities=["audio"],
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    target_transform = lambda tensor: TextRegressionDataset.normalize_targets(tensor.to(torch.float32))
    val_dataset = AudioWaveLMDataset(
        val_dataset_msp,
        feature_extractor,
        max_seconds=None,
        do_resample=True,
        label_key="val_arousal",
        label_dtype=torch.float32,
        label_transform=target_transform,
        include_attention_mask=True,
        extractor_padding=True,
        extractor_truncation=True,
        extractor_max_length=160000,
    )

    tester = AudioRegressionTester()
    results = tester.test(val_dataset, checkpoint_dir, batch_size=8, output_dir=output_dir, title="WavLM Audio Regression")

    print(f"Loss: {results['avg_loss']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R2: {results['r2']:.4f}")
    print(f"CCC Mean: {results['ccc_mean']:.4f}")


if __name__ == "__main__":
    main()