"""Deterministic synthetic MRI-style image-mask generator from the supplied notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import SEED


def make_blob_mask(height: int = 64, width: int = 64, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    cx = int(rng.integers(16, 48))
    cy = int(rng.integers(16, 48))
    rx = int(rng.integers(6, 16))
    ry = int(rng.integers(6, 16))
    mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1
    return mask.astype(np.float32)


def make_medical_sample(seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Create one 64×64 grayscale sample and an elliptical binary target mask."""
    rng = np.random.default_rng(seed)
    image = rng.normal(0.35, 0.07, size=(64, 64))
    mask = make_blob_mask(64, 64, seed)
    image += mask * rng.uniform(0.25, 0.50)
    image = np.clip(image, 0.0, 1.0)
    return image[..., None].astype(np.float32), mask[..., None].astype(np.float32)


def generate_synthetic_dataset(
    num_samples: int = 2500,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index in range(num_samples):
        image, mask = make_medical_sample(seed + index)
        images.append(image)
        masks.append(mask)
    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.float32)


def split_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the notebook's 70% / 15% / 15% split."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        images, masks, test_size=0.30, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=seed
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def save_demo_samples(
    image_dir: Path,
    mask_dir: Path,
    count: int = 8,
    seed: int = SEED + 5000,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image, mask = make_medical_sample(seed + index)
        image_png = Image.fromarray(np.clip(image.squeeze() * 255, 0, 255).astype(np.uint8), mode="L")
        mask_png = Image.fromarray((mask.squeeze() * 255).astype(np.uint8), mode="L")
        name = f"sample_{index + 1:02d}.png"
        image_png.save(image_dir / name)
        mask_png.save(mask_dir / name)
