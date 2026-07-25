"""Segmentation-mask preprocessing and postprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .config import IMAGE_SIZE, MASK_THRESHOLD
from .image_preprocessing import load_image


def preprocess_mask(
    mask: Any,
    image_size: tuple[int, int] = IMAGE_SIZE,
    threshold: float = MASK_THRESHOLD,
) -> np.ndarray:
    """Resize a mask with nearest-neighbor interpolation and binarize it."""
    pil_mask = load_image(mask).convert("L")
    resized = pil_mask.resize((image_size[1], image_size[0]), Image.Resampling.NEAREST)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    binary = (array >= threshold).astype(np.float32)
    return binary[None, ..., None]


def postprocess_probability_map(
    probability: np.ndarray,
    original_size: tuple[int, int],
    threshold: float = MASK_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore a model probability map and binary mask to the input resolution."""
    prob = np.asarray(probability, dtype=np.float32).squeeze()
    if prob.ndim != 2:
        raise ValueError(f"Expected a 2D probability map; received shape {prob.shape}.")
    prob_uint8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    prob_image = Image.fromarray(prob_uint8, mode="L").resize(
        original_size, Image.Resampling.BILINEAR
    )
    restored_probability = np.asarray(prob_image, dtype=np.float32) / 255.0
    restored_mask = (restored_probability >= threshold).astype(np.uint8)
    return restored_probability, restored_mask
