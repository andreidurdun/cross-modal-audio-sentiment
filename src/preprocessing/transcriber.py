"""Audio transcription helpers powered by Faster-Whisper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None

from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps


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

    def _build_batch_tokenizer(self, language: Optional[str]) -> Tokenizer:
        resolved_language = language or self.config.default_language
        if not self.model.model.is_multilingual:
            resolved_language = "en"
        elif resolved_language is None:
            raise ValueError("language must be provided for multilingual batched transcription")

        return Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=resolved_language,
        )

    def _build_batch_options(self) -> TranscriptionOptions:
        return TranscriptionOptions(
            beam_size=self.config.beam_size,
            best_of=self.config.beam_size,
            patience=1,
            length_penalty=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True,
            prompt_reset_on_temperature=0.5,
            temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            initial_prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=[-1],
            without_timestamps=True,
            max_initial_timestamp=1.0,
            word_timestamps=False,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",
            multilingual=False,
            max_new_tokens=None,
            clip_timestamps="0",
            hallucination_silence_threshold=None,
            hotwords=None,
        )

    def _collect_chunk_features(self, audio_paths: List[str]) -> tuple[list[list[str]], list[np.ndarray], list[dict]]:
        transcripts_by_file: list[list[str]] = [[] for _ in audio_paths]
        all_features: list[np.ndarray] = []
        all_metadata: list[dict] = []
        sampling_rate = self.model.feature_extractor.sampling_rate
        chunk_length = self.model.feature_extractor.chunk_length

        vad_parameters = VadOptions(
            max_speech_duration_s=chunk_length,
            min_silence_duration_ms=160,
        )

        for file_index, audio_path in enumerate(audio_paths):
            audio = decode_audio(audio_path, sampling_rate=sampling_rate)
            duration = audio.shape[0] / sampling_rate if audio.shape[0] else 0.0

            if self.config.vad_filter:
                clip_timestamps = get_speech_timestamps(audio, vad_parameters)
            elif duration < chunk_length:
                clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
            else:
                clip_timestamps = [{"start": 0, "end": audio.shape[0]}]

            if not clip_timestamps:
                continue

            audio_chunks, chunks_metadata = collect_chunks(
                audio,
                clip_timestamps,
                max_duration=chunk_length,
            )

            for chunk, chunk_metadata in zip(audio_chunks, chunks_metadata):
                feature = self.model.feature_extractor(chunk)[..., :-1]
                all_features.append(pad_or_trim(feature))
                all_metadata.append(
                    {
                        "file_index": file_index,
                        "offset": chunk_metadata["offset"],
                        "duration": chunk_metadata["duration"],
                    }
                )

        return transcripts_by_file, all_features, all_metadata

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

        transcripts_by_file, all_features, all_metadata = self._collect_chunk_features(audio_paths)
        if not all_features:
            return ["" for _ in audio_paths]

        tokenizer = self._build_batch_tokenizer(language)
        options = self._build_batch_options()

        for start_index in range(0, len(all_features), effective_batch_size):
            feature_batch = np.stack(all_features[start_index : start_index + effective_batch_size])
            metadata_batch = all_metadata[start_index : start_index + effective_batch_size]
            segmented_outputs = self.batched_model.forward(
                feature_batch,
                tokenizer,
                metadata_batch,
                options,
            )

            for chunk_metadata, chunk_segments in zip(metadata_batch, segmented_outputs):
                chunk_text = " ".join(segment["text"].strip() for segment in chunk_segments).strip()
                if chunk_text:
                    transcripts_by_file[chunk_metadata["file_index"]].append(chunk_text)

        return [" ".join(parts).strip() for parts in transcripts_by_file]
