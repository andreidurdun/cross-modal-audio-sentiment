"""Audio preprocessing utilities: resampling, mono conversion, augmentations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torchaudio


@dataclass
class AudioProcessorConfig:
    target_sample_rate: int = 16_000
    force_mono: bool = True
    apply_telephony_augmentation: bool = False


class AudioProcessor:
    """Loads waveforms from disk and applies standard preprocessing steps."""

    def __init__(self, config: Optional[AudioProcessorConfig] = None) -> None:
        self.config = config or AudioProcessorConfig()
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        self._downsampler = torchaudio.transforms.Resample(
            orig_freq=self.config.target_sample_rate,
            new_freq=8_000,
        )
        self._upsampler = torchaudio.transforms.Resample(
            orig_freq=8_000,
            new_freq=self.config.target_sample_rate,
        )

    def load_waveform(self, path: str) -> torch.Tensor:
        """Loads an audio file and returns a mono waveform at the target rate."""
        waveform, sample_rate = torchaudio.load(path)

        if self.config.force_mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._resample(waveform, sample_rate)

        if self.config.apply_telephony_augmentation:
            waveform = self.telephony_augmentation(waveform)

        return waveform.squeeze(0)

    def telephony_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simulates a narrow-band telephone channel."""
        degraded = self._downsampler(waveform)
        restored = self._upsampler(degraded)
        return restored

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.config.target_sample_rate:
            return waveform

        if sample_rate not in self._resamplers:
            self._resamplers[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.config.target_sample_rate,
            )
        return self._resamplers[sample_rate](waveform)
