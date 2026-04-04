from pathlib import Path

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from transformers import AutoTokenizer

from src.data.dataset import MSP_Podcast_Dataset
from src.data.text_datasets import TextRegressionDataset
from src.utils import TextRegressionTester


def main():
    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_de_json = data_dir / "Transcription_de.json"
    checkpoint_dir = Path("checkpoints/roberta_text_de_regression")
    output_dir = Path("results/roberta_text_de_regression")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_de_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_de_json}")

    print("=" * 80)
    print("Loading MSP-Podcast German Text Data for Regression Testing")
    print("=" * 80)

    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_de_json=str(transcripts_de_json),
        partition="Development",
        modalities=["text_de"],
    )

    tester = TextRegressionTester()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "best_model")
    val_dataset = TextRegressionDataset(val_dataset_msp, tokenizer, text_fields=["text_de", "text_en"], max_length=128, padding=True)
    results = tester.test(val_dataset, checkpoint_dir, batch_size=32, output_dir=output_dir, title="RoBERTa Text DE Regression")

    print(f"Loss: {results['avg_loss']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R2: {results['r2']:.4f}")
    print(f"CCC Mean: {results['ccc_mean']:.4f}")


if __name__ == "__main__":
    main()
