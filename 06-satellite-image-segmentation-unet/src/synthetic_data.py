from __future__ import annotations

import numpy as np


def make_rect_mask(height: int = 64, width: int = 64, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(rng.integers(2, 6))):
        x1 = int(rng.integers(2, width - 20))
        y1 = int(rng.integers(2, height - 20))
        box_width = int(rng.integers(6, 18))
        box_height = int(rng.integers(6, 18))
        mask[y1:y1 + box_height, x1:x1 + box_width] = 1.0
    return mask


def make_satellite_sample(seed: int, image_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    image[..., 1] = rng.normal(0.45, 0.08, size=(image_size, image_size))
    image[..., 0] = rng.normal(0.35, 0.06, size=(image_size, image_size))
    image[..., 2] = rng.normal(0.30, 0.05, size=(image_size, image_size))
    mask = make_rect_mask(image_size, image_size, seed)
    for channel in range(3):
        image[..., channel] += mask * float(rng.uniform(0.25, 0.5))
    return np.clip(image, 0.0, 1.0).astype(np.float32), mask[..., None].astype(np.float32)


def generate_dataset(count: int = 2500, seed: int = 42, image_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    samples = [make_satellite_sample(seed + index, image_size) for index in range(count)]
    images = np.stack([item[0] for item in samples])
    masks = np.stack([item[1] for item in samples])
    return images, masks
