from pathlib import Path
from typing import Optional
import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from src.data.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset
from src.models import load_ccmt_only_model
from src.utils.regression_trainers import compute_regression_metrics


SUPPORTED_MODALITIES = ["text_en", "text_es", "text_de", "text_fr", "audio"]
SUPPORTED_PARTITIONS = ["train", "val", "test1"]


def parse_modalities(modalities_arg: Optional[str]) -> list[str]:
    if not modalities_arg:
        return ["text_en", "text_es", "audio"]

    modalities = [item.strip() for item in modalities_arg.split(",") if item.strip()]
    invalid_modalities = [item for item in modalities if item not in SUPPORTED_MODALITIES]
    if invalid_modalities:
        raise ValueError(
            f"Modalitati invalide: {invalid_modalities}. Alege dintre {SUPPORTED_MODALITIES}"
        )
    return modalities


def resolve_effective_modalities(requested_modalities: Optional[list[str]], saved_model_config: dict) -> list[str]:
    saved_modalities = saved_model_config.get("modalities")
    if saved_modalities:
        saved_modalities = list(saved_modalities)
        if requested_modalities is not None and requested_modalities != saved_modalities:
            raise ValueError(
                "Modalitatile cerute pentru test nu corespund cu checkpoint-ul. "
                f"Checkpoint-ul foloseste {saved_modalities}, dar ai cerut {requested_modalities}. "
                "Pentru testare pe embeddings trebuie folosite exact modalitatile din checkpoint "
                "sau un alt checkpoint antrenat pe acea combinatie."
            )
        return saved_modalities

    if requested_modalities is not None:
        return requested_modalities

    return ["text_en", "text_es", "audio"]


def build_modality_suffix(modalities: list[str]) -> str:
    return "_".join(modalities)


def infer_embeddings_dir_from_checkpoint(checkpoint_dir: Path, modalities: list[str]) -> Optional[Path]:
    checkpoint_parts = checkpoint_dir.resolve().parts
    if "regression_pipeline" not in checkpoint_parts:
        return None

    pipeline_index = checkpoint_parts.index("regression_pipeline")
    if pipeline_index + 1 >= len(checkpoint_parts):
        return None

    run_name = checkpoint_parts[pipeline_index + 1]
    return (
        PROJECT_ROOT
        / "MSP_Podcast"
        / "regression_pipeline"
        / run_name
        / f"embeddings_{build_modality_suffix(modalities)}"
    )


def resolve_embeddings_dir(
    embeddings_dir_arg: Optional[str],
    modalities: list[str],
    checkpoint_dir: Optional[Path] = None,
) -> Path:
    if embeddings_dir_arg:
        return Path(embeddings_dir_arg)

    if checkpoint_dir is not None:
        inferred_dir = infer_embeddings_dir_from_checkpoint(checkpoint_dir, modalities)
        if inferred_dir is not None:
            return inferred_dir

    if modalities == ["text_en", "text_es", "audio"]:
        return PROJECT_ROOT / "MSP_Podcast" / "embeddings"

    return PROJECT_ROOT / "MSP_Podcast" / f"embeddings_{build_modality_suffix(modalities)}"


def collect_embedding_inputs(batch: dict, modalities: list[str], device: str) -> dict[str, torch.Tensor]:
    return {
        f"{modality}_emb": batch[f"{modality}_emb"].to(device, non_blocking=True)
        for modality in modalities
    }


class CCMTRegressionTester:
    def __init__(self, modalities: list[str], device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.modalities = modalities
        self.loss_fn = nn.MSELoss()

    def _build_model_config(self, saved_model_config: dict, dataset: PrecomputedEmbeddingsDataset) -> dict:
        embedding_dims = dataset.get_embedding_dims()
        return {
            "num_outputs": int(saved_model_config.get("num_outputs", 2)),
            "text_en_dim": int(saved_model_config.get("text_en_dim", embedding_dims.get("text_en", 768))),
            "text_es_dim": int(saved_model_config.get("text_es_dim", embedding_dims.get("text_es", 768))),
            "text_de_dim": int(saved_model_config.get("text_de_dim", embedding_dims.get("text_de", 768))),
            "text_fr_dim": int(saved_model_config.get("text_fr_dim", embedding_dims.get("text_fr", 768))),
            "audio_dim": int(saved_model_config.get("audio_dim", embedding_dims.get("audio", 768))),
            "ccmt_dim": int(saved_model_config.get("ccmt_dim", 768)),
            "num_patches_per_modality": int(saved_model_config.get("num_patches_per_modality", 100)),
            "ccmt_depth": int(saved_model_config.get("ccmt_depth", 4)),
            "ccmt_heads": int(saved_model_config.get("ccmt_heads", 4)),
            "ccmt_mlp_dim": int(saved_model_config.get("ccmt_mlp_dim", 1024)),
            "ccmt_dropout": float(saved_model_config.get("ccmt_dropout", 0.1)),
            "modalities": list(saved_model_config.get("modalities", self.modalities)),
        }

    def load_model(self, checkpoint_dir: Path, model_config: dict):
        model = load_ccmt_only_model(
            device=self.device,
            text_en_dim=model_config["text_en_dim"],
            text_es_dim=model_config["text_es_dim"],
            text_de_dim=model_config["text_de_dim"],
            text_fr_dim=model_config["text_fr_dim"],
            audio_dim=model_config["audio_dim"],
            num_classes=model_config["num_outputs"],
            ccmt_dim=model_config["ccmt_dim"],
            num_patches_per_modality=model_config["num_patches_per_modality"],
            ccmt_depth=model_config["ccmt_depth"],
            ccmt_heads=model_config["ccmt_heads"],
            ccmt_mlp_dim=model_config["ccmt_mlp_dim"],
            ccmt_dropout=model_config["ccmt_dropout"],
            modalities=model_config["modalities"],
        )

        model_path = checkpoint_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device)

    @torch.no_grad()
    def evaluate(self, model, data_loader: DataLoader) -> dict:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            embedding_inputs = collect_embedding_inputs(batch, self.modalities, self.device)
            labels = batch["val_arousal"].to(self.device, dtype=torch.float32, non_blocking=True)
            predictions = model(**embedding_inputs).float()
            loss = self.loss_fn(predictions, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

        y_pred = PrecomputedEmbeddingsDataset.denormalize_val_arousal(torch.cat(all_predictions, dim=0)).numpy()
        y_true = PrecomputedEmbeddingsDataset.denormalize_val_arousal(torch.cat(all_labels, dim=0)).numpy()
        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["avg_loss"] = float(total_loss / max(total_samples, 1))
        metrics["rmse_valence"] = float(np.sqrt(metrics["mse_valence"]))
        metrics["rmse_arousal"] = float(np.sqrt(metrics["mse_arousal"]))
        metrics["num_samples"] = int(total_samples)
        metrics["predictions"] = y_pred.tolist()
        metrics["labels"] = y_true.tolist()
        return metrics

    def test(
        self,
        dataset: PrecomputedEmbeddingsDataset,
        checkpoint_dir: Path,
        output_dir: Path,
        batch_size: int,
        partition: str,
        saved_model_config: dict,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_config = self._build_model_config(saved_model_config, dataset)
        self.modalities = list(model_config["modalities"])
        model = self.load_model(checkpoint_dir, model_config)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = self.evaluate(model, data_loader)

        results_to_save = {key: value for key, value in results.items() if key not in ["predictions", "labels"]}
        results_to_save["partition"] = partition
        results_to_save["modalities"] = self.modalities
        results_to_save["checkpoint_dir"] = str(checkpoint_dir)

        results_json_path = output_dir / "test_results.json"
        with results_json_path.open("w", encoding="utf-8") as file_handle:
            json.dump(results_to_save, file_handle, indent=2)

        predictions_path = output_dir / "predictions.json"
        with predictions_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "partition": partition,
                    "predictions": results["predictions"],
                    "labels": results["labels"],
                },
                file_handle,
                indent=2,
            )

        print(f"[OK] Results saved: {results_json_path}")
        print(f"[OK] Predictions saved: {predictions_path}")
        return results_to_save


def main():
    parser = argparse.ArgumentParser(description="Test CCMT regression checkpoints")
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Lista de modalitati separate prin virgula. Daca lipseste, se citeste din training_config.json sau se foloseste text_en,text_es,audio.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directorul checkpoint-ului CCMT de evaluat.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Directorul cu embeddings precompute.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Director pentru rezultate.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="test1",
        choices=SUPPORTED_PARTITIONS,
        help="Split-ul pe care se face evaluarea.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size pentru evaluare.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints/ccmt_multimodal_regression0")
    training_config_path = checkpoint_dir / "training_config.json"
    saved_model_config = {}
    saved_batch_size = None
    if training_config_path.exists():
        with training_config_path.open("r", encoding="utf-8") as file_handle:
            saved_config = json.load(file_handle)
        saved_model_config = saved_config.get("model_architecture", {})
        saved_hyperparameters = saved_config.get("training_hyperparameters", {})
        saved_batch_size = saved_hyperparameters.get("batch_size")

    requested_modalities = parse_modalities(args.modalities) if args.modalities else None
    modalities = resolve_effective_modalities(requested_modalities, saved_model_config)
    modality_suffix = build_modality_suffix(modalities)
    embeddings_dir = resolve_embeddings_dir(args.embeddings_dir, modalities, checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"ccmt_multimodal_regression_{modality_suffix}"
    batch_size = int(saved_batch_size) if saved_batch_size is not None else int(args.batch_size)

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    print("=" * 80)
    print("Loading Precomputed Embeddings for CCMT Regression")
    print("=" * 80)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Embeddings dir: {embeddings_dir}")
    print(f"Partition: {args.partition}")
    print(f"Modalities: {modalities}")
    print(f"Batch size: {batch_size}")

    dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(embeddings_dir),
        partition=args.partition,
        regression=True,
        modalities=modalities,
    )

    tester = CCMTRegressionTester(modalities=modalities)
    results = tester.test(
        dataset=dataset,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        partition=args.partition,
        saved_model_config=saved_model_config,
    )

    print("\n" + "=" * 80)
    print("REGRESSION RESULTS")
    print("=" * 80)
    print(f"Loss: {results['avg_loss']:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"MSE valence: {results['mse_valence']:.4f}")
    print(f"MSE arousal: {results['mse_arousal']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"RMSE valence: {results['rmse_valence']:.4f}")
    print(f"RMSE arousal: {results['rmse_arousal']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"MAE valence: {results['mae_valence']:.4f}")
    print(f"MAE arousal: {results['mae_arousal']:.4f}")
    print(f"R2: {results['r2']:.4f}")
    print(f"R2 valence: {results['r2_valence']:.4f}")
    print(f"R2 arousal: {results['r2_arousal']:.4f}")
    print(f"CCC valence: {results['ccc_valence']:.4f}")
    print(f"CCC arousal: {results['ccc_arousal']:.4f}")
    print(f"CCC mean: {results['ccc_mean']:.4f}")
    print(f"Pearson valence: {results['pearson_valence']:.4f}")
    print(f"Pearson arousal: {results['pearson_arousal']:.4f}")
    print(f"Pearson mean: {results['pearson_mean']:.4f}")


if __name__ == "__main__":
    main()