"""Audio transcription helpers powered by Faster-Whisper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from faster_whisper import WhisperModel


@dataclass
class TranscriberConfig:
    model_size: str = "distil-medium.en"
    device: str = "cuda"
    compute_type: str = "float16"
    default_language: Optional[str] = "en"
    beam_size: int = 5
    vad_filter: bool = True


class AudioTranscriber:
    def __init__(self, config: Optional[TranscriberConfig] = None) -> None:
        self.config = config or TranscriberConfig()
        self.model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=self.config.beam_size,
            language=language or self.config.default_language,
            vad_filter=self.config.vad_filter,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        return [self.transcribe(path) for path in audio_paths]
