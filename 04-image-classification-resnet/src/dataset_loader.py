from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_cifar100(validation_size: int = 10_000, normalize: bool = True) -> DatasetBundle:
    """Load CIFAR-100 using the same final-slice validation strategy as the notebook."""
    if validation_size <= 0 or validation_size >= 50_000:
        raise ValueError("validation_size must be between 1 and 49,999.")
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - exercised only in full environment
        raise RuntimeError("TensorFlow is required to download CIFAR-100.") from exc

    (x_full, y_full), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    y_full = y_full.reshape(-1)
    y_test = y_test.reshape(-1)
    split = len(x_full) - validation_size
    x_train, x_validation = x_full[:split], x_full[split:]
    y_train, y_validation = y_full[:split], y_full[split:]

    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_validation = x_validation.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

    return DatasetBundle(x_train, y_train, x_validation, y_validation, x_test, y_test)


def one_hot_labels(bundle: DatasetBundle, num_classes: int = 100) -> DatasetBundle:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for one-hot encoding.") from exc
    return DatasetBundle(
        bundle.x_train,
        tf.keras.utils.to_categorical(bundle.y_train, num_classes),
        bundle.x_validation,
        tf.keras.utils.to_categorical(bundle.y_validation, num_classes),
        bundle.x_test,
        tf.keras.utils.to_categorical(bundle.y_test, num_classes),
    )
