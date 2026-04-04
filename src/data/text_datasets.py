from typing import Iterable

import torch
from torch.utils.data import Dataset

from src.data.dataset import MSP_Podcast_Dataset


class TextEncoderDataset(Dataset):
    """Reusable text dataset with pre-tokenization for transformer classifiers."""

    def __init__(
        self,
        msp_dataset: MSP_Podcast_Dataset,
        tokenizer,
        text_fields: Iterable[str],
        max_length: int = 128,
        padding: str | bool = "max_length",
        validate_labels: bool = False,
        sanitize_input_ids: bool = False,
    ):
        self.msp_dataset = msp_dataset
        self.tokenizer = tokenizer
        self.text_fields = list(text_fields)
        self.max_length = max_length
        self.padding = padding
        self.validate_labels = validate_labels
        self.sanitize_input_ids = sanitize_input_ids

        self.labels = self._extract_labels()
        self.encodings = self._precompute_encodings()

    def _extract_labels(self) -> list[int]:
        labels: list[int] = []
        for _, row in self.msp_dataset.metadata.iterrows():
            label = int(row["label_id"])
            if self.validate_labels and label not in [0, 1, 2]:
                raise ValueError(f"CRITICAL: Found invalid label '{label}'. Must be 0, 1, or 2.")
            labels.append(label)
        return labels

    def _resolve_text(self, sample: dict) -> str:
        for field in self.text_fields:
            text = sample.get(field, "")
            if text and text.strip():
                return text.strip()
        return "[EMPTY]"

    def _precompute_encodings(self):
        texts = [self._resolve_text(self.msp_dataset[idx]) for idx in range(len(self.msp_dataset))]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if self.sanitize_input_ids:
            self._sanitize_input_ids(encodings)
        return encodings

    def _sanitize_input_ids(self, encodings) -> None:
        input_ids = encodings["input_ids"]
        vocab_size = int(self.tokenizer.vocab_size)
        bad_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if bad_mask.any():
            unk_id = self.tokenizer.unk_token_id
            if unk_id is None:
                unk_id = 0
            input_ids = input_ids.clone()
            input_ids[bad_mask] = int(unk_id)
            encodings["input_ids"] = input_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TextRegressionDataset(Dataset):
    """Reusable text dataset with normalized valence/arousal targets."""

    MIN_TARGET_VALUE = 1.0
    MAX_TARGET_VALUE = 7.0

    def __init__(
        self,
        msp_dataset: MSP_Podcast_Dataset,
        tokenizer,
        text_fields: Iterable[str],
        max_length: int = 128,
        padding: str | bool = "max_length",
        sanitize_input_ids: bool = False,
    ):
        self.msp_dataset = msp_dataset
        self.tokenizer = tokenizer
        self.text_fields = list(text_fields)
        self.max_length = max_length
        self.padding = padding
        self.sanitize_input_ids = sanitize_input_ids

        self.targets = self._extract_targets()
        self.encodings = self._precompute_encodings()

    @classmethod
    def normalize_targets(cls, targets: torch.Tensor) -> torch.Tensor:
        return (targets - cls.MIN_TARGET_VALUE) / (cls.MAX_TARGET_VALUE - cls.MIN_TARGET_VALUE)

    @classmethod
    def denormalize_targets(cls, targets: torch.Tensor) -> torch.Tensor:
        return targets * (cls.MAX_TARGET_VALUE - cls.MIN_TARGET_VALUE) + cls.MIN_TARGET_VALUE

    def _extract_targets(self) -> torch.Tensor:
        targets: list[list[float]] = []
        for _, row in self.msp_dataset.metadata.iterrows():
            targets.append([float(row["EmoVal"]), float(row["EmoAct"])])
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        return self.normalize_targets(target_tensor)

    def _resolve_text(self, sample: dict) -> str:
        for field in self.text_fields:
            text = sample.get(field, "")
            if text and text.strip():
                return text.strip()
        return "[EMPTY]"

    def _precompute_encodings(self):
        texts = [self._resolve_text(self.msp_dataset[idx]) for idx in range(len(self.msp_dataset))]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if self.sanitize_input_ids:
            self._sanitize_input_ids(encodings)
        return encodings

    def _sanitize_input_ids(self, encodings) -> None:
        input_ids = encodings["input_ids"]
        vocab_size = int(self.tokenizer.vocab_size)
        bad_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if bad_mask.any():
            unk_id = self.tokenizer.unk_token_id
            if unk_id is None:
                unk_id = 0
            input_ids = input_ids.clone()
            input_ids[bad_mask] = int(unk_id)
            encodings["input_ids"] = input_ids

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = self.targets[idx].clone().detach()
        return item
