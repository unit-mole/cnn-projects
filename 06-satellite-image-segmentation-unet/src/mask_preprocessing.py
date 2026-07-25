from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def load_mask(source: Any) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            mask = source.copy()
        elif isinstance(source, (str, Path)):
            mask = Image.open(source)
        elif isinstance(source, np.ndarray):
            array = np.asarray(source)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            mask = Image.fromarray(array)
        else:
            raise TypeError("Mask must be a PIL image, NumPy array, or filesystem path.")
        return ImageOps.exif_transpose(mask).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The mask could not be decoded.") from exc


def preprocess_binary_mask(
    source: Any,
    size: tuple[int, int] = (64, 64),
    threshold: int = 127,
) -> np.ndarray:
    mask = load_mask(source)
    resized = mask.resize((size[1], size[0]), Image.Resampling.NEAREST)
    array = np.asarray(resized, dtype=np.uint8)
    return (array > threshold).astype(np.float32)
