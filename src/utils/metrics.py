from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    id2label: Dict[int, str],
    total_loss: float,
    num_batches: int,
) -> dict:
    """Compute a consistent set of classification metrics used across evaluators."""
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision_macro = precision_score(labels, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)

    f1_per_class = {}
    for label_id, label_name in id2label.items():
        f1_per_class[label_name] = f1_score(
            labels,
            predictions,
            labels=[label_id],
            average="micro",
            zero_division=0,
        )

    cm = confusion_matrix(labels, predictions)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_per_class": {k: float(v) for k, v in f1_per_class.items()},
        "confusion_matrix": cm.tolist(),
        "avg_loss": float(total_loss / max(num_batches, 1)),
        "num_samples": int(len(labels)),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }
