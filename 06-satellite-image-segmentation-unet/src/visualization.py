from __future__ import annotations

import numpy as np
from PIL import Image


def binary_mask_to_image(mask: np.ndarray, output_size: tuple[int, int] | None = None) -> Image.Image:
    array = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    image = Image.fromarray(array, mode="L")
    if output_size is not None:
        image = image.resize(output_size, Image.Resampling.NEAREST)
    return image


def probability_to_image(probability: np.ndarray, output_size: tuple[int, int] | None = None) -> Image.Image:
    prob = np.clip(np.asarray(probability).squeeze(), 0.0, 1.0)
    # A simple blue-to-red map implemented without an additional plotting dependency.
    red = np.clip(2.0 * prob, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - prob), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * prob - 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    if output_size is not None:
        image = image.resize(output_size, Image.Resampling.BILINEAR)
    return image


def create_overlay(original: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = original.convert("RGB")
    mask_image = binary_mask_to_image(mask, base.size)
    color_layer = Image.new("RGB", base.size, (255, 80, 40))
    transparent = Image.new("RGB", base.size, (0, 0, 0))
    colored_mask = Image.composite(color_layer, transparent, mask_image)
    return Image.blend(base, Image.blend(base, colored_mask, 0.75), alpha)
