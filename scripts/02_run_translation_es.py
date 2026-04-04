"""Batch translation entry point (Optimized for GPU Batching)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import sys
try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

# Am actualizat importul pentru noul translator
from src.preprocessing.translator import NLLBTranslator, NLLBTranslatorConfig
from src.utils.helpers import set_seed

# Poți adăuga asta în funcția ta de parse_args, dar o definesc aici ca referință
BATCH_SIZE = 32  # 32 sau 64 este ideal pentru RTX 4060 8GB în FP16
INPUT_PATH = Path("MSP_Podcast/Transcription_en_cache.json")
OUTPUT_PATH = Path("MSP_Podcast/Transcription_es.json")
SEED = 42

def main() -> None:
    set_seed(SEED)

    print(f"Loading transcripts from {INPUT_PATH}...")
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        transcripts = json.load(f)

    # Inițializăm noul translator (folosește NLLB și FP16 automat)
    translator = NLLBTranslator()
    
    # Separam cheile și textele pentru a putea face batching
    keys = list(transcripts.keys())
    texts = list(transcripts.values())
    translated = {}

    total_items = len(keys)
    print(f"Starting translation of {total_items} items in batches of {BATCH_SIZE}...")
    
    start_time = time.time()

    # Iterăm prin listă din BATCH_SIZE în BATCH_SIZE
    for i in range(0, total_items, BATCH_SIZE):
        batch_keys = keys[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]
        
        # Apelăm NOUA metodă de batching pe care am creat-o
        batch_translated = translator.translate_batch(batch_texts)
        
        # Reconstruim dicționarul mapând cheile originale cu textele traduse
        for key, translated_text in zip(batch_keys, batch_translated):
            translated[key] = translated_text
            
        # Un mic log pentru a vedea progresul (opțional, dar foarte util)
        if (i + BATCH_SIZE) % (BATCH_SIZE * 10) == 0 or i + BATCH_SIZE >= total_items:
            current_count = min(i + BATCH_SIZE, total_items)
            print(f"Progress: {current_count}/{total_items} ({(current_count/total_items)*100:.1f}%)")

    elapsed_time = time.time() - start_time
    print(f"Translation completed in {elapsed_time:.2f} seconds!")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    print(f"Saved translated transcripts to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()