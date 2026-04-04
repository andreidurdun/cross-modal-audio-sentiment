import json
from pathlib import Path
from typing import Any


def load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_training_config(section: str, config_path: Path) -> dict[str, Any]:
    config = load_json_config(config_path)
    section_config = config.get(section)
    if section_config is None:
        raise KeyError(f"Missing training config section '{section}' in {config_path}")
    if not isinstance(section_config, dict):
        raise TypeError(f"Training config section '{section}' must be an object")
    return section_config
