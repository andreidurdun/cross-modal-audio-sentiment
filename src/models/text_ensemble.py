"""Text sentiment ensemble that combines EN and RO experts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing.translator import EnglishToRomanianTranslator


@dataclass
class SentimentModelConfig:
    english_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    romanian_model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


class SentimentEnsemble:
    def __init__(
        self,
        translator: Optional[EnglishToRomanianTranslator] = None,
        config: Optional[SentimentModelConfig] = None,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config or SentimentModelConfig()

        self.translator = translator or EnglishToRomanianTranslator()

        self.tokenizer_en = AutoTokenizer.from_pretrained(self.config.english_model_name)
        self.model_en = AutoModelForSequenceClassification.from_pretrained(
            self.config.english_model_name
        ).to(self.device)

        self.tokenizer_ro = AutoTokenizer.from_pretrained(self.config.romanian_model_name)
        self.model_ro = AutoModelForSequenceClassification.from_pretrained(
            self.config.romanian_model_name
        ).to(self.device)

        self.labels = ["unsatisfied", "neutral", "satisfied"]

    def _get_probs(self, text: str, tokenizer, model) -> torch.Tensor:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()[0]

    def predict(self, text_en: str):
        probs_en = self._get_probs(text_en, self.tokenizer_en, self.model_en)
        translation = self.translator.translate(text_en)
        probs_ro = self._get_probs(translation, self.tokenizer_ro, self.model_ro)

        final_probs = (probs_en + probs_ro) / 2
        final_class_idx = final_probs.argmax()

        return {
            "original_en": text_en,
            "translated_ro": translation,
            "probs_en": probs_en,
            "probs_ro": probs_ro,
            "ensemble_probs": final_probs,
            "final_label": self.labels[final_class_idx],
        }
