"""Training-time preprocessing, augmentation, and imbalance utilities."""

from __future__ import annotations

import numpy as np


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    return images / 255.0 if images.max(initial=0.0) > 1.0 else images


def one_hot_encode(labels: np.ndarray, number_of_classes: int = 2) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if np.any(labels < 0) or np.any(labels >= number_of_classes):
        raise ValueError("Labels fall outside the requested class range.")
    return np.eye(number_of_classes, dtype=np.float32)[labels]


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Return balanced inverse-frequency weights without requiring scikit-learn."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {int(label): float(total / (len(classes) * count)) for label, count in zip(classes, counts)}


def build_safe_augmentation(seed: int = 42):
    """Build subtle transformations suitable for preserving discriminative details."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required to build augmentation layers.") from exc

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.06, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.10, seed=seed + 2),
        ],
        name="safe_fine_grained_augmentation",
    )
