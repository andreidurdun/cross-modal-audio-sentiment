"""Audio transcription helpers powered by Faster-Whisper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None


@dataclass
class TranscriberConfig:
    model_size: str = "distil-medium.en"
    device: str = "cuda"
    compute_type: str = "float16"
    default_language: Optional[str] = "en"
    beam_size: int = 5
    vad_filter: bool = True
    batch_size: int = 8


class AudioTranscriber:
    def __init__(self, config: Optional[TranscriberConfig] = None) -> None:
        self.config = config or TranscriberConfig()
        self.model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        self.batched_model = BatchedInferencePipeline(model=self.model) if BatchedInferencePipeline else None

    @staticmethod
    def _segments_to_text(segments) -> str:
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=self.config.beam_size,
            language=language or self.config.default_language,
            vad_filter=self.config.vad_filter,
        )
        return self._segments_to_text(segments)

    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        if not audio_paths:
            return []

        effective_batch_size = batch_size or self.config.batch_size
        if self.batched_model is None or effective_batch_size <= 1:
            return [self.transcribe(path, language=language) for path in audio_paths]

        transcripts: List[str] = []
        for audio_path in audio_paths:
            segments, _ = self.batched_model.transcribe(
                audio_path,
                beam_size=self.config.beam_size,
                language=language or self.config.default_language,
                vad_filter=self.config.vad_filter,
                batch_size=effective_batch_size,
            )
            transcripts.append(self._segments_to_text(segments))
        return transcripts
