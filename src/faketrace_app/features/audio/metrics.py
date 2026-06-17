from __future__ import annotations

from collections.abc import Iterable

import numpy as np


KNOWN_AUDIO_TYPES = ("speech", "sound", "singing", "music")


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    eps = 1e-12
    scores = []
    for class_id in range(num_classes):
        true_positive = np.logical_and(y_true == class_id, y_pred == class_id).sum()
        false_positive = np.logical_and(y_true != class_id, y_pred == class_id).sum()
        false_negative = np.logical_and(y_true == class_id, y_pred != class_id).sum()
        precision = true_positive / (true_positive + false_positive + eps)
        recall = true_positive / (true_positive + false_negative + eps)
        scores.append(2 * precision * recall / (precision + recall + eps))
    return float(np.mean(scores))


def type_macro_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_types: Iterable[str | None],
    num_classes: int,
) -> dict[str, float]:
    sample_types_array = np.asarray(list(sample_types), dtype=object)
    metrics: dict[str, float] = {}
    present_scores = []
    for audio_type in KNOWN_AUDIO_TYPES:
        mask = sample_types_array == audio_type
        if not np.any(mask):
            continue
        score = macro_f1_score(y_true[mask], y_pred[mask], num_classes=num_classes)
        metrics[f"{audio_type}_macro_f1"] = score
        present_scores.append(score)
    if present_scores:
        metrics["track2_macro_f1"] = float(np.mean(present_scores))
    return metrics


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    sample_types: Iterable[str | None] | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1_score(y_true, y_pred, num_classes=num_classes),
    }
    if sample_types is not None:
        metrics.update(
            type_macro_f1_scores(
                y_true,
                y_pred,
                sample_types=sample_types,
                num_classes=num_classes,
            )
        )
    return metrics
