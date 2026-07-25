from __future__ import annotations

import numpy as np


def iou_boxes_np(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    x_a = np.maximum(y_true[:, 0], y_pred[:, 0])
    y_a = np.maximum(y_true[:, 1], y_pred[:, 1])
    x_b = np.minimum(y_true[:, 2], y_pred[:, 2])
    y_b = np.minimum(y_true[:, 3], y_pred[:, 3])

    intersection = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)
    true_area = np.maximum(0, y_true[:, 2] - y_true[:, 0]) * np.maximum(0, y_true[:, 3] - y_true[:, 1])
    pred_area = np.maximum(0, y_pred[:, 2] - y_pred[:, 0]) * np.maximum(0, y_pred[:, 3] - y_pred[:, 1])
    union = true_area + pred_area - intersection + 1e-6
    return intersection / union
