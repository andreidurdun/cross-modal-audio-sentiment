"""
Dataset Loader pentru embeddings pre-calculate.
Permite training rapid folosind embeddings salvate pe disc.
"""
import csv
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PrecomputedEmbeddingsDataset(Dataset):
    """
    Dataset pentru embeddings pre-calculate.
    Încarcă embeddings de pe disc pentru training rapid.
    """

    SUPPORTED_MODALITIES = ["text_en", "text_es", "text_de", "text_fr", "audio"]

    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        partition: str = "train",
        device: str = "cpu",
        regression: bool = False,
        modalities: List[str] | None = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.partition = partition
        self.device = device
        self.regression = regression

        embeddings_file = self.embeddings_dir / f"embeddings_{partition}.pt"
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_file}\n"
                f"Run 'python scripts/extract_and_save_embeddings.py --partition {partition}' first!"
            )

        print(f"Loading embeddings from {embeddings_file}...")
        self.data = torch.load(embeddings_file, map_location="cpu")
        self.labels = self.data["labels"]
        self.file_ids = self.data["file_ids"]
        self.metadata = self.data.get("metadata", {})
        self.available_modalities = [
            modality for modality in self.SUPPORTED_MODALITIES if modality in self.data
        ]

        if modalities is None:
            modalities = [
                modality for modality in ["text_en", "text_es", "audio"]
                if modality in self.available_modalities
            ]

        invalid_modalities = set(modalities) - set(self.SUPPORTED_MODALITIES)
        if invalid_modalities:
            raise ValueError(
                f"Modalitatile {sorted(invalid_modalities)} nu sunt suportate. "
                f"Alege dintre: {self.SUPPORTED_MODALITIES}"
            )

        missing_modalities = set(modalities) - set(self.available_modalities)
        if missing_modalities:
            raise ValueError(
                f"Embeddings file {embeddings_file} nu contine modalitatile {sorted(missing_modalities)}. "
                f"Modalitati disponibile: {self.available_modalities}"
            )

        if not modalities:
            raise ValueError("Trebuie selectata cel putin o modalitate pentru PrecomputedEmbeddingsDataset")

        self.modalities = list(modalities)
        self.embeddings = {modality: self.data[modality] for modality in self.modalities}

        self.valence = self.data.get("valence", None)
        self.arousal = self.data.get("arousal", None)

        if self.regression:
            if self.valence is None or self.arousal is None:
                self._recover_regression_targets(embeddings_file)
            self.valence = self.valence.to(torch.float32)
            self.arousal = self.arousal.to(torch.float32)

            min_val = 1.0
            max_val = 7.0
            self.valence = (self.valence - min_val) / (max_val - min_val)
            self.arousal = (self.arousal - min_val) / (max_val - min_val)

        print(f"Loaded {len(self)} samples from {partition}")
        print(f"  Modalitati active: {self.modalities}")
        for modality in self.modalities:
            print(f"  {modality} embedding dim: {self.embeddings[modality].shape[-1]}")

    def _recover_regression_targets(self, embeddings_file: Path) -> None:
        labels_csv = PROJECT_ROOT / "MSP_Podcast" / "Labels" / "labels_consensus.csv"
        if not labels_csv.exists():
            raise ValueError(
                "Embeddings file must contain 'valence' and 'arousal' tensors for regression task, "
                f"iar fisierul de labels nu exista: {labels_csv}"
            )

        label_lookup: dict[str, tuple[float, float]] = {}
        with labels_csv.open("r", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                file_name = str(row["FileName"])
                values = (float(row["EmoVal"]), float(row["EmoAct"]))
                label_lookup[file_name] = values
                if file_name.endswith(".wav"):
                    label_lookup[file_name[:-4]] = values

        valence = []
        arousal = []
        for file_id in self.file_ids:
            key = str(file_id)
            if key not in label_lookup:
                raise KeyError(f"Nu am gasit {file_id} in {labels_csv}")
            emo_val, emo_act = label_lookup[key]
            valence.append(emo_val)
            arousal.append(emo_act)

        self.valence = torch.tensor(valence, dtype=torch.float32)
        self.arousal = torch.tensor(arousal, dtype=torch.float32)
        self.data["valence"] = self.valence.clone()
        self.data["arousal"] = self.arousal.clone()
        torch.save(self.data, embeddings_file)
        print(f"✓ Added missing valence/arousal targets to {embeddings_file}")

    @staticmethod
    def denormalize_val_arousal(val_arousal_norm: torch.Tensor) -> torch.Tensor:
        min_val = 1.0
        max_val = 7.0
        return val_arousal_norm * (max_val - min_val) + min_val

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            "labels": self.labels[idx],
            "file_id": self.file_ids[idx],
        }
        for modality in self.modalities:
            batch[f"{modality}_emb"] = self.embeddings[modality][idx]
        if self.regression:
            batch["val_arousal"] = torch.stack([self.valence[idx], self.arousal[idx]])
        return batch

    def get_embedding_dims(self) -> Dict[str, int]:
        return {
            modality: self.data[modality].shape[-1]
            for modality in self.available_modalities
        }


def create_dataloaders(
    embeddings_dir: str = "data/embeddings",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    modalities: List[str] | None = None,
) -> Dict[str, DataLoader]:
    dataloaders = {}

    for partition in ["train", "val", "test1"]:
        try:
            dataset = PrecomputedEmbeddingsDataset(
                embeddings_dir,
                partition,
                modalities=modalities,
            )
            dataloaders[partition] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(partition == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            print(
                f"Created DataLoader for {partition}: {len(dataset)} samples, "
                f"{len(dataloaders[partition])} batches"
            )
        except FileNotFoundError as exc:
            print(f"Skipping {partition}: {exc}")

    return dataloaders
