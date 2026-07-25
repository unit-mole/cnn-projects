"""Segmentation metrics shared by training and evaluation code."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_batch(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ..., None]
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[None, ...]
        else:
            arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"Expected a 2D, 3D, or 4D mask array; received shape {arr.shape}.")
    return arr


def dice_coefficient_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Return the mean Dice coefficient across a batch."""
    true = _as_batch(y_true).reshape(len(_as_batch(y_true)), -1)
    pred = _as_batch(y_pred).reshape(len(_as_batch(y_pred)), -1)
    if true.shape != pred.shape:
        raise ValueError(f"Mask shapes must match. Received {true.shape} and {pred.shape}.")
    intersection = np.sum(true * pred, axis=1)
    score = (2.0 * intersection + smooth) / (
        np.sum(true, axis=1) + np.sum(pred, axis=1) + smooth
    )
    return float(np.mean(score))


def iou_score_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Return the mean intersection-over-union across a batch."""
    true = _as_batch(y_true).reshape(len(_as_batch(y_true)), -1)
    pred = _as_batch(y_pred).reshape(len(_as_batch(y_pred)), -1)
    if true.shape != pred.shape:
        raise ValueError(f"Mask shapes must match. Received {true.shape} and {pred.shape}.")
    intersection = np.sum(true * pred, axis=1)
    union = np.sum(true, axis=1) + np.sum(pred, axis=1) - intersection
    score = (intersection + smooth) / (union + smooth)
    return float(np.mean(score))


def binary_pixel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute pixel-level precision, recall, F1, accuracy, FP rate, and FN rate."""
    true = (_as_batch(y_true) >= 0.5).astype(np.uint8)
    pred = (_as_batch(y_pred) >= 0.5).astype(np.uint8)
    if true.shape != pred.shape:
        raise ValueError(f"Mask shapes must match. Received {true.shape} and {pred.shape}.")

    tp = int(np.sum((true == 1) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    total = max(tp + tn + fp + fn, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "pixel_accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_pixel_rate": fp / total,
        "false_negative_pixel_rate": fn / total,
    }


def dice_coef_tf(y_true: Any, y_pred: Any, smooth: float = 1e-6) -> Any:
    """TensorFlow soft Dice metric with the original saved-model function name."""
    import tensorflow as tf

    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    return tf.reduce_mean(
        (2.0 * intersection + smooth)
        / (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)
    )


def iou_tf(y_true: Any, y_pred: Any, smooth: float = 1e-6) -> Any:
    """TensorFlow soft IoU metric with the original saved-model function name."""
    import tensorflow as tf

    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = (
        tf.reduce_sum(y_true_f, axis=1)
        + tf.reduce_sum(y_pred_f, axis=1)
        - intersection
    )
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


def get_custom_objects() -> dict[str, Any]:
    return {"dice_coef_tf": dice_coef_tf, "iou_tf": iou_tf}
