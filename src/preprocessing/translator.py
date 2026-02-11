"""Translation utilities for the text sentiment channel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class TranslatorConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-ro"
    max_length: int = 512


class EnglishToRomanianTranslator:
    """Thin wrapper around the Helsinki-NLP Marian models."""

    def __init__(self, config: Optional[TranslatorConfig] = None) -> None:
        self.config = config or TranslatorConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name).to(self.device)
        self.model.eval()

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
