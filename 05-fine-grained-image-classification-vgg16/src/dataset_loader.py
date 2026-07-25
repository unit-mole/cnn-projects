"""Dataset loader for the actual supplied CIFAR-10 cat-versus-dog experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SEED
from .data_preprocessing import normalize_pixels

CAT_SOURCE_LABEL = 3
DOG_SOURCE_LABEL = 5


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _filter_and_remap(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    mask = np.isin(labels, [CAT_SOURCE_LABEL, DOG_SOURCE_LABEL])
    filtered_images = images[mask]
    filtered_labels = np.where(labels[mask] == CAT_SOURCE_LABEL, 0, 1).astype(np.int64)
    return filtered_images, filtered_labels


def load_cifar10_cat_dog(
    validation_size: int = 2_000,
    *,
    stratified: bool = True,
    seed: int = SEED,
) -> DatasetSplits:
    """Load CIFAR-10 and retain only cats and dogs.

    The original notebook used the final 2,000 filtered training examples as
    validation data. The modular default uses a reproducible stratified split
    to remove ordering dependence while preserving the same 8k/2k/2k sizes.
    """
    try:
        from tensorflow.keras.datasets import cifar10
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required to download CIFAR-10.") from exc

    (x_pool, y_pool), (x_test, y_test) = cifar10.load_data()
    x_pool, y_pool = _filter_and_remap(x_pool, y_pool)
    x_test, y_test = _filter_and_remap(x_test, y_test)

    if validation_size <= 0 or validation_size >= len(x_pool):
        raise ValueError("validation_size must be between 1 and the pool size minus 1.")

    if stratified:
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scikit-learn is required for stratified splitting.") from exc
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_pool,
            y_pool,
            test_size=validation_size,
            random_state=seed,
            stratify=y_pool,
        )
    else:
        split_at = len(x_pool) - validation_size
        x_train, y_train = x_pool[:split_at], y_pool[:split_at]
        x_validation, y_validation = x_pool[split_at:], y_pool[split_at:]

    return DatasetSplits(
        x_train=normalize_pixels(x_train),
        y_train=y_train,
        x_validation=normalize_pixels(x_validation),
        y_validation=y_validation,
        x_test=normalize_pixels(x_test),
        y_test=y_test,
    )
