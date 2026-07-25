"""Evaluation, baseline comparison, threshold analysis, and error metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASELINE_THRESHOLD, MASK_THRESHOLD
from .metrics import binary_pixel_metrics, dice_coefficient_np, iou_score_np


def intensity_threshold_baseline(images: np.ndarray, threshold: float = BASELINE_THRESHOLD) -> np.ndarray:
    return (np.asarray(images, dtype=np.float32) > threshold).astype(np.float32)


def predict_probabilities(model: object, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
    probabilities = model.predict(images, batch_size=batch_size, verbose=0)
    return np.asarray(probabilities, dtype=np.float32)


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = MASK_THRESHOLD,
) -> dict[str, float]:
    hard_masks = (np.asarray(probabilities) >= threshold).astype(np.float32)
    metrics = {
        "soft_dice": dice_coefficient_np(y_true, probabilities),
        "soft_iou": iou_score_np(y_true, probabilities),
        "hard_dice": dice_coefficient_np(y_true, hard_masks),
        "hard_iou": iou_score_np(y_true, hard_masks),
        "threshold": float(threshold),
    }
    metrics.update(binary_pixel_metrics(y_true, hard_masks))
    return metrics


def per_sample_scores(
    y_true: np.ndarray,
    hard_masks: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for index in range(len(y_true)):
        rows.append(
            {
                "sample_index": index,
                "dice": dice_coefficient_np(y_true[index:index + 1], hard_masks[index:index + 1]),
                "iou": iou_score_np(y_true[index:index + 1], hard_masks[index:index + 1]),
                "true_positive_fraction": float(np.mean(y_true[index])),
                "predicted_positive_fraction": float(np.mean(hard_masks[index])),
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.20, 0.85, 0.05)
    rows = []
    for threshold in thresholds:
        hard_masks = (probabilities >= threshold).astype(np.float32)
        rows.append(
            {
                "threshold": float(threshold),
                "dice": dice_coefficient_np(y_true, hard_masks),
                "iou": iou_score_np(y_true, hard_masks),
            }
        )
    return pd.DataFrame(rows)
