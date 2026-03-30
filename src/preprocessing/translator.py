"""Translation utilities for the text sentiment channel (Optimized for RTX 4060)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass

class NLLBTranslatorConfig:
    # 600M este rapid și bun. Poți încerca și "facebook/nllb-200-distilled-1.3B" dacă ai VRAM liber
    model_name: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "eng_Latn"   # Limba sursă: Engleză
    tgt_lang: str = "spa_Latn"   # Limba țintă: Spaniolă
    max_length: int = 512


class NLLBTranslatorConfigFR(NLLBTranslatorConfig):
    """Config pentru traducere engleză-franceză cu NLLB."""
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", src_lang: str = "eng_Latn", max_length: int = 512):
        super().__init__(model_name=model_name, src_lang=src_lang, tgt_lang="fra_Latn", max_length=max_length)


class NLLBTranslatorConfigDE(NLLBTranslatorConfig):
    """Config pentru traducere engleză-germană cu NLLB."""
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", src_lang: str = "eng_Latn", max_length: int = 512):
        super().__init__(model_name=model_name, src_lang=src_lang, tgt_lang="deu_Latn", max_length=max_length)



class NLLBTranslator:
    """Optimized wrapper around Meta's NLLB model for RTX 4000 series."""

    def __init__(self, config: Optional[NLLBTranslatorConfig] = None) -> None:
        self.config = config or NLLBTranslatorConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # OPTIMIZAREA 1: FP16 (Obligatoriu pentru RTX 4060 pentru viteză maximă)
        self.compute_dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading {self.config.model_name} on {self.device} in {self.compute_dtype}...")

        # NLLB necesită specificarea limbii sursă direct în tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            src_lang=self.config.src_lang
        )

        # Încărcăm modelul direct în formatul optimizat
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            dtype=self.compute_dtype
        ).to(self.device)

        self.model.eval() # Modul de inferență (dezactivează dropout-ul)

        # NLLB necesită ID-ul limbii țintă la momentul generării
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.config.tgt_lang)

    def translate(self, text: str) -> str:
        """Traduce un singur text (Folosește translate_batch pentru seturi de date mari)."""
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
                forced_bos_token_id=self.tgt_lang_id, # Specific NLLB
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        OPTIMIZAREA 2: Traducere în Batch.
        Aceasta este metoda pe care trebuie să o folosești pentru a procesa
        transcripturile din MSP-Podcast rapid pe RTX 4060.
        """
        if not texts:
            return []

        # Tokenizăm o listă întreagă. Adăugăm padding pentru a egala lungimea tensorilor.
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,       # Obligatoriu pentru batching
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                forced_bos_token_id=self.tgt_lang_id,
            )

        # Decodificăm tot batch-ul deodată
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Pipeline pentru traducere engleză-germană
# Pipeline pentru traducere engleză-germană
class NLLBTranslatorDE(NLLBTranslator):
    """Wrapper pentru traducere engleză-germană cu NLLB."""
    def __init__(self, config: Optional[NLLBTranslatorConfigDE] = None) -> None:
        super().__init__(config or NLLBTranslatorConfigDE())


# Pipeline pentru traducere engleză-franceză
class NLLBTranslatorFR(NLLBTranslator):
    """Wrapper pentru traducere engleză-franceză cu NLLB."""
    def __init__(self, config: Optional[NLLBTranslatorConfigFR] = None) -> None:
        super().__init__(config or NLLBTranslatorConfigFR())