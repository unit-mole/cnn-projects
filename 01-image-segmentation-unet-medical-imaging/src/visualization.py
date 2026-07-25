"""Visualization helpers for masks, overlays, probability maps, and error maps."""

from __future__ import annotations

import numpy as np
from PIL import Image


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    binary = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def probability_to_heatmap(probability: np.ndarray) -> Image.Image:
    """Create a compact blue-to-red probability map without an external colormap dependency."""
    prob = np.clip(np.asarray(probability, dtype=np.float32).squeeze(), 0.0, 1.0)
    red = np.clip(2.0 * prob, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - prob), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * prob - 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def create_overlay(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.42,
) -> Image.Image:
    base = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert("RGB")
    binary = np.asarray(mask).squeeze() > 0
    if binary.shape != (base.height, base.width):
        binary_image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L").resize(
            base.size, Image.Resampling.NEAREST
        )
        binary = np.asarray(binary_image) > 0
    base_array = np.asarray(base, dtype=np.float32)
    overlay_color = np.zeros_like(base_array)
    overlay_color[..., 0] = 255.0
    overlay_color[..., 1] = 72.0
    overlay_color[..., 2] = 72.0
    output = base_array.copy()
    output[binary] = (1.0 - alpha) * base_array[binary] + alpha * overlay_color[binary]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def create_error_map(y_true: np.ndarray, y_pred: np.ndarray) -> Image.Image:
    """Green=true positive, red=false positive, blue=false negative."""
    true = np.asarray(y_true).squeeze() > 0
    pred = np.asarray(y_pred).squeeze() > 0
    rgb = np.zeros((*true.shape, 3), dtype=np.uint8)
    rgb[true & pred] = [64, 196, 99]
    rgb[~true & pred] = [230, 65, 65]
    rgb[true & ~pred] = [68, 112, 230]
    return Image.fromarray(rgb, mode="RGB")
