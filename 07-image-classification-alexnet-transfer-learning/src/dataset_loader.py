from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .class_mapping import CIFAR10_CLASSES, validate_class_names
from .data_preprocessing import build_augmentation


@dataclass
class DatasetBundle:
    train: object
    validation: object
    test: object
    class_names: list[str]
    train_labels: np.ndarray
    test_labels: np.ndarray


def _prepare_array_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    training: bool,
    augment: bool,
    seed: int,
):
    import tensorflow as tf

    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(min(len(images), 20_000), seed=seed, reshuffle_each_iteration=True)

    augmenter = build_augmentation(seed) if training and augment else None

    def preprocess(image, label):
        image = tf.image.resize(tf.cast(image, tf.float32), (image_size, image_size)) / 255.0
        if augmenter is not None:
            image = augmenter(image, training=True)
        return image, label

    return (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_cifar10(
    image_size: int = 227,
    batch_size: int = 64,
    validation_fraction: float = 0.15,
    seed: int = 42,
    augment: bool = True,
) -> DatasetBundle:
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    train_idx, validation_idx = train_test_split(
        np.arange(len(x_train)),
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train,
    )

    train_ds = _prepare_array_dataset(
        x_train[train_idx], y_train[train_idx], image_size, batch_size, True, augment, seed
    )
    validation_ds = _prepare_array_dataset(
        x_train[validation_idx], y_train[validation_idx], image_size, batch_size, False, False, seed
    )
    test_ds = _prepare_array_dataset(
        x_test, y_test, image_size, batch_size, False, False, seed
    )

    return DatasetBundle(
        train=train_ds,
        validation=validation_ds,
        test=test_ds,
        class_names=list(CIFAR10_CLASSES),
        train_labels=y_train[train_idx],
        test_labels=y_test,
    )


def load_folder_dataset(
    data_dir: str | Path,
    image_size: int = 227,
    batch_size: int = 32,
    validation_fraction: float = 0.15,
    seed: int = 42,
    augment: bool = True,
) -> DatasetBundle:
    """Load a folder dataset with one subdirectory per class.

    The function uses a training/validation split. A dedicated test directory is
    preferred; if absent, validation is also returned as the test set and the
    README warns users to create a real held-out test set for final reporting.
    """
    import tensorflow as tf

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    train_dir = data_dir / "train" if (data_dir / "train").is_dir() else data_dir
    test_dir = data_dir / "test"

    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_fraction,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
    )
    validation_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_fraction,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
    )
    class_names = validate_class_names(train_raw.class_names)

    augmenter = build_augmentation(seed) if augment else None

    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        if augmenter is not None:
            image = augmenter(image, training=True)
        return image, label

    def normalize_only(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    train_ds = train_raw.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_raw.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    if test_dir.is_dir():
        test_raw = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            shuffle=False,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            label_mode="int",
            class_names=class_names,
        )
        test_ds = test_raw.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    else:
        test_ds = validation_ds

    train_labels = np.concatenate([y.numpy().reshape(-1) for _, y in train_raw], axis=0)
    test_labels = np.concatenate([y.numpy().reshape(-1) for _, y in (test_raw if test_dir.is_dir() else validation_raw)], axis=0)

    return DatasetBundle(train_ds, validation_ds, test_ds, class_names, train_labels, test_labels)
