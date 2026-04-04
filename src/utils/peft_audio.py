from __future__ import annotations

from pathlib import Path
from typing import Optional

from peft import PeftConfig, PeftModel
from transformers import AutoModelForAudioClassification


def load_peft_audio_classification_model(
    checkpoint_path: str | Path,
    *,
    num_labels: Optional[int] = None,
    problem_type: Optional[str] = None,
):
    checkpoint_path = Path(checkpoint_path)
    peft_config = PeftConfig.from_pretrained(checkpoint_path)

    base_model_kwargs = {
        "ignore_mismatched_sizes": True,
    }
    if peft_config.base_model_name_or_path:
        base_model_kwargs["pretrained_model_name_or_path"] = peft_config.base_model_name_or_path
    if getattr(peft_config, "revision", None):
        base_model_kwargs["revision"] = peft_config.revision
    if num_labels is not None:
        base_model_kwargs["num_labels"] = num_labels
    if problem_type is not None:
        base_model_kwargs["problem_type"] = problem_type

    base_model_name = base_model_kwargs.pop("pretrained_model_name_or_path")
    base_model = AutoModelForAudioClassification.from_pretrained(base_model_name, **base_model_kwargs)
    return PeftModel.from_pretrained(base_model, checkpoint_path)
