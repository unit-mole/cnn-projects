from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def build_augmentation(seed: int = 42):
    """Create conservative image augmentations suitable for CIFAR-10."""
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=seed + 1),
            tf.keras.layers.RandomZoom(0.08, fill_mode="reflect", seed=seed + 2),
            tf.keras.layers.RandomContrast(0.10, seed=seed + 3),
        ],
        name="safe_augmentation",
    )


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights, strict=True)}
