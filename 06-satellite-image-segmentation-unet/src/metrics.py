from __future__ import annotations

import numpy as np


def _flatten_batch(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[None, ...]
    if arr.ndim < 3:
        raise ValueError("Expected a mask or batch of masks with at least two spatial dimensions.")
    return arr.reshape(arr.shape[0], -1)


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    true_flat = _flatten_batch(y_true)
    pred_flat = _flatten_batch(y_pred)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(f"Mask shape mismatch: {true_flat.shape} vs {pred_flat.shape}")
    intersection = np.sum(true_flat * pred_flat, axis=1)
    score = (2.0 * intersection + smooth) / (
        np.sum(true_flat, axis=1) + np.sum(pred_flat, axis=1) + smooth
    )
    return float(np.mean(score))


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    true_flat = _flatten_batch(y_true)
    pred_flat = _flatten_batch(y_pred)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(f"Mask shape mismatch: {true_flat.shape} vs {pred_flat.shape}")
    intersection = np.sum(true_flat * pred_flat, axis=1)
    union = np.sum(true_flat, axis=1) + np.sum(pred_flat, axis=1) - intersection
    return float(np.mean((intersection + smooth) / (union + smooth)))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> dict[str, float]:
    true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred).astype(bool)
    if true.shape != pred.shape:
        raise ValueError(f"Mask shape mismatch: {true.shape} vs {pred.shape}")
    tp = float(np.logical_and(true, pred).sum())
    fp = float(np.logical_and(~true, pred).sum())
    fn = float(np.logical_and(true, ~pred).sum())
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
    return {"precision": precision, "recall": recall, "f1": f1}


def dice_coef_tf(y_true, y_pred, smooth: float = 1e-6):
    import tensorflow as tf
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    return tf.reduce_mean(
        (2.0 * intersection + smooth)
        / (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)
    )


def iou_tf(y_true, y_pred, smooth: float = 1e-6):
    import tensorflow as tf
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))
