from __future__ import annotations

import numpy as np

from .metrics import dice_coefficient, iou_score, precision_recall_f1


def evaluate_binary_masks(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predicted = (np.asarray(probabilities) >= threshold).astype(np.float32)
    true = (np.asarray(y_true) >= 0.5).astype(np.float32)
    results = {
        "dice": dice_coefficient(true, predicted),
        "iou": iou_score(true, predicted),
        "pixel_accuracy": float(np.mean(true == predicted)),
    }
    results.update(precision_recall_f1(true, predicted))
    return results
