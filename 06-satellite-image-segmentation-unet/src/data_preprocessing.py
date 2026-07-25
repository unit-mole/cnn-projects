from __future__ import annotations

import numpy as np


def augment_pair(image: np.ndarray, mask: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same safe spatial transform to an image and its mask."""
    rng = np.random.default_rng(seed)
    image_out = np.asarray(image).copy()
    mask_out = np.asarray(mask).copy()

    if rng.random() < 0.5:
        image_out = np.flip(image_out, axis=1)
        mask_out = np.flip(mask_out, axis=1)
    if rng.random() < 0.5:
        image_out = np.flip(image_out, axis=0)
        mask_out = np.flip(mask_out, axis=0)

    rotations = int(rng.integers(0, 4))
    image_out = np.rot90(image_out, k=rotations, axes=(0, 1))
    mask_out = np.rot90(mask_out, k=rotations, axes=(0, 1))
    return np.ascontiguousarray(image_out), np.ascontiguousarray(mask_out)


def normalize_images(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.dtype == np.uint8 or float(array.max(initial=0)) > 1.0:
        array = array.astype(np.float32) / 255.0
    return np.clip(array.astype(np.float32), 0.0, 1.0)
