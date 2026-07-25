from __future__ import annotations

import numpy as np
import tensorflow as tf


def make_detection_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 12_000,
    canvas_size: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place one resized MNIST digit at a random location on each canvas."""
    if n_samples > len(images):
        raise ValueError("n_samples cannot exceed the available source images.")

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(images), size=n_samples, replace=False)
    x_values, class_values, boxes = [], [], []

    for index in selected:
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        digit = images[index].astype(np.float32) / 255.0
        label = int(labels[index])

        scale = rng.uniform(0.8, 1.4)
        new_size = int(round(28 * scale))
        resized = tf.image.resize(digit[..., None], (new_size, new_size)).numpy().squeeze()

        x1 = int(rng.integers(0, canvas_size - new_size + 1))
        y1 = int(rng.integers(0, canvas_size - new_size + 1))
        x2, y2 = x1 + new_size, y1 + new_size
        canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], resized)

        x_values.append(canvas[..., None])
        class_values.append(label)
        boxes.append([x1 / canvas_size, y1 / canvas_size, x2 / canvas_size, y2 / canvas_size])

    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(class_values, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32),
    )
