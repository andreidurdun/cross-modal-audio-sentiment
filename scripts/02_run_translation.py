"""Batch translation entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocessing.translator import EnglishToRomanianTranslator
from src.utils.helpers import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate transcripts EN -> RO")
    parser.add_argument("--input", type=Path, default=Path("data/transcripts/transcripts_en.json"))
    parser.add_argument("--output", type=Path, default=Path("data/cache/translations_en_ro.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with args.input.open("r", encoding="utf-8") as f:
        transcripts = json.load(f)

    translator = EnglishToRomanianTranslator()
    translated = {
        key: translator.translate(text)
        for key, text in transcripts.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
