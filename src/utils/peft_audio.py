from __future__ import annotations

from pathlib import Path
from typing import Optional

from peft import PeftConfig, PeftModel
from transformers import AutoModelForAudioClassification


def _ensure_audio_regression_head_shape(base_model, num_labels: Optional[int]) -> None:
    if num_labels is None:
        return

    classifier = getattr(base_model, "classifier", None)
    if isinstance(classifier, type(getattr(base_model, "classifier", None))) and hasattr(classifier, "out_features"):
        if classifier.out_features != num_labels and hasattr(classifier, "in_features"):
            base_model.classifier = classifier.__class__(classifier.in_features, num_labels).to(
                device=classifier.weight.device,
                dtype=classifier.weight.dtype,
            )

    base_model.num_labels = num_labels
    base_model.config.num_labels = num_labels


def load_peft_audio_classification_model(
    checkpoint_path: str | Path,
    *,
    num_labels: Optional[int] = None,
    problem_type: Optional[str] = None,
):
    checkpoint_path = Path(checkpoint_path)
    adapter_config_path = checkpoint_path / "adapter_config.json"

    if not adapter_config_path.exists():
        model_kwargs = {
            "ignore_mismatched_sizes": True,
        }
        if num_labels is not None:
            model_kwargs["num_labels"] = num_labels
        if problem_type is not None:
            model_kwargs["problem_type"] = problem_type

        model = AutoModelForAudioClassification.from_pretrained(checkpoint_path, **model_kwargs)
        _ensure_audio_regression_head_shape(model, num_labels)
        return model

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
    _ensure_audio_regression_head_shape(base_model, num_labels)
    return PeftModel.from_pretrained(base_model, checkpoint_path)
