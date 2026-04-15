"""Translation utilities for the text sentiment channel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass

class NLLBTranslatorConfig:
   
    model_name: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "eng_Latn"   # Limba sursa: Engleza
    tgt_lang: str = "spa_Latn"   # Limba tinta: Spaniola
    max_length: int = 512


class NLLBTranslatorConfigFR(NLLBTranslatorConfig):
    """Config pentru traducere engleza-franceza cu NLLB."""
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", src_lang: str = "eng_Latn", max_length: int = 512):
        super().__init__(model_name=model_name, src_lang=src_lang, tgt_lang="fra_Latn", max_length=max_length)


class NLLBTranslatorConfigDE(NLLBTranslatorConfig):
    """Config pentru traducere engleza-germana cu NLLB."""
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", src_lang: str = "eng_Latn", max_length: int = 512):
        super().__init__(model_name=model_name, src_lang=src_lang, tgt_lang="deu_Latn", max_length=max_length)



class NLLBTranslator:
    """Optimized wrapper around Meta's NLLB model for RTX 4000 series."""

    def __init__(self, config: Optional[NLLBTranslatorConfig] = None) -> None:
        self.config = config or NLLBTranslatorConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # FP16 pe GPU pentru viteza
        self.compute_dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading {self.config.model_name} on {self.device} in {self.compute_dtype}...")

        # NLLB necesita specificarea limbii sursa direct in tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            src_lang=self.config.src_lang
        )

        # Incarcam modelul direct in formatul optimizat
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            dtype=self.compute_dtype
        ).to(self.device)

        self.model.eval() # Modul de inferenta (dezactiveaza dropout-ul)

        # NLLB necesita ID-ul limbii tinta la momentul generarii
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.config.tgt_lang)

    def translate(self, text: str) -> str:
        """Traduce un singur text (Foloseste translate_batch pentru seturi de date mari)."""
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
        """Traducere in batch, recomandat pentru seturi mari de date."""
        if not texts:
            return []

        # Tokenizam o lista intreaga. Adaugam padding pentru a egala lungimea tensorilor.
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

        # Decodificam tot batch-ul deodata
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Pipeline pentru traducere engleza-germana
# Pipeline pentru traducere engleza-germana
class NLLBTranslatorDE(NLLBTranslator):
    """Wrapper pentru traducere engleza-germana cu NLLB."""
    def __init__(self, config: Optional[NLLBTranslatorConfigDE] = None) -> None:
        super().__init__(config or NLLBTranslatorConfigDE())


# Pipeline pentru traducere engleza-franceza
class NLLBTranslatorFR(NLLBTranslator):
    """Wrapper pentru traducere engleza-franceza cu NLLB."""
    def __init__(self, config: Optional[NLLBTranslatorConfigFR] = None) -> None:
        super().__init__(config or NLLBTranslatorConfigFR())