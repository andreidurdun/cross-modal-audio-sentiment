"""Batch translation entry point."""
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


from src.preprocessing.translator import NLLBTranslator, NLLBTranslatorConfig
from src.utils.helpers import set_seed

BATCH_SIZE = 32 
INPUT_PATH = Path("MSP_Podcast/Transcription_en.json")
OUTPUT_PATH = Path("MSP_Podcast/Transcription_es.json")
SEED = 42

def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English transcripts to Spanish")
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH, help="Input English transcripts JSON")
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH, help="Output Spanish transcripts JSON")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for translation")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading transcripts from {args.input_path}...")
    with args.input_path.open("r", encoding="utf-8") as f:
        transcripts = json.load(f)


    translator = NLLBTranslator()
    

    keys = list(transcripts.keys())
    texts = list(transcripts.values())
    translated = {}

    total_items = len(keys)
    print(f"Starting translation of {total_items} items in batches of {args.batch_size}...")
    
    start_time = time.time()


    for i in range(0, total_items, args.batch_size):
        batch_keys = keys[i : i + args.batch_size]
        batch_texts = texts[i : i + args.batch_size]
        
        batch_translated = translator.translate_batch(batch_texts)
        
        #reconstruim dictionarul mapand cheile originale cu textele traduse
        for key, translated_text in zip(batch_keys, batch_translated):
            translated[key] = translated_text
            
   
        if (i + args.batch_size) % (args.batch_size * 10) == 0 or i + args.batch_size >= total_items:
            current_count = min(i + args.batch_size, total_items)
            print(f"Progress: {current_count}/{total_items} ({(current_count/total_items)*100:.1f}%)")

    elapsed_time = time.time() - start_time
    print(f"Translation completed in {elapsed_time:.2f} seconds!")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    print(f"Saved translated transcripts to {args.output_path}")

if __name__ == "__main__":
    main()