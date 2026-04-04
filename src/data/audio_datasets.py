from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import MSP_Podcast_Dataset


class AudioWaveLMDataset(Dataset):
    """Reusable audio dataset wrapper for WavLM training and evaluation."""

    def __init__(
        self,
        msp_dataset: MSP_Podcast_Dataset,
        feature_extractor,
        sample_rate: int = 16000,
        max_seconds: Optional[int] = 5,
        do_resample: bool = False,
        label_key: str = "label_id",
        label_dtype: torch.dtype = torch.long,
        label_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        include_attention_mask: bool = False,
        extractor_padding: bool = False,
        extractor_truncation: bool = False,
        extractor_max_length: Optional[int] = None,
    ):
        self.msp_dataset = msp_dataset
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.do_resample = do_resample
        self.label_key = label_key
        self.label_dtype = label_dtype
        self.label_transform = label_transform
        self.include_attention_mask = include_attention_mask
        self.extractor_padding = extractor_padding
        self.extractor_truncation = extractor_truncation
        self.extractor_max_length = extractor_max_length

    def __len__(self) -> int:
        return len(self.msp_dataset)

    def _get_audio(self, sample: dict) -> np.ndarray:
        audio = sample["audio"]
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        if np.isnan(audio).any() or np.isinf(audio).any():
            audio = np.zeros_like(audio)

        if self.do_resample and sample.get("sample_rate", self.sample_rate) != self.sample_rate:
            import librosa

            audio = librosa.resample(
                audio,
                orig_sr=sample.get("sample_rate", self.sample_rate),
                target_sr=self.sample_rate,
            )

        if self.max_seconds is not None:
            max_samples = self.sample_rate * self.max_seconds
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]

        return audio

    def __getitem__(self, idx: int) -> dict:
        sample = self.msp_dataset[idx]
        audio = self._get_audio(sample)

        extractor_kwargs = {
            "sampling_rate": self.sample_rate,
            "return_tensors": "pt",
        }
        if self.extractor_padding:
            extractor_kwargs["padding"] = True
        if self.extractor_truncation:
            extractor_kwargs["truncation"] = True
        if self.extractor_max_length is not None:
            extractor_kwargs["max_length"] = self.extractor_max_length

        inputs = self.feature_extractor(audio, **extractor_kwargs)

        label = sample.get(self.label_key)
        if label is None:
            label = sample.get("label_id", sample.get("label"))
        if label is None:
            raise KeyError(f"Missing label for sample index {idx}. Tried keys: '{self.label_key}', 'label_id', 'label'.")

        if isinstance(label, torch.Tensor):
            label_tensor = label.clone().detach().to(dtype=self.label_dtype)
        else:
            label_tensor = torch.tensor(label, dtype=self.label_dtype)

        if self.label_transform is not None:
            label_tensor = self.label_transform(label_tensor)

        batch = {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": label_tensor,
        }

        if self.include_attention_mask:
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(inputs["input_values"].shape[-1], dtype=torch.long)
            else:
                attention_mask = attention_mask.squeeze(0)
            batch["attention_mask"] = attention_mask

        return batch


class AudioCollator:
    def __call__(self, batch: list[dict]) -> dict:
        input_features = [sample["input_values"] for sample in batch]
        labels = torch.stack([sample["labels"] for sample in batch])

        batch_size = len(input_features)
        max_length = max(feature.shape[-1] for feature in input_features)

        padded_inputs = torch.zeros((batch_size, max_length), dtype=torch.float32)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        for i, feature in enumerate(input_features):
            seq_len = feature.shape[-1]
            padded_inputs[i, :seq_len] = feature
            attention_mask[i, :seq_len] = 1

        return {
            "input_values": padded_inputs,
            "attention_mask": attention_mask,
            "labels": labels,
        }
