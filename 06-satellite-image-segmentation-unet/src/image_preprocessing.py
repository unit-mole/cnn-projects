from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def load_rgb_image(source: Any) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported image extension: {path.suffix}")
            image = Image.open(path)
        elif isinstance(source, np.ndarray):
            array = np.asarray(source)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array)
        else:
            raise TypeError("Image must be a PIL image, NumPy array, or filesystem path.")
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file could not be decoded as a supported image.") from exc


def preprocess_image(source: Any, size: tuple[int, int] = (64, 64)) -> tuple[np.ndarray, Image.Image]:
    image = load_rgb_image(source)
    resized = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if array.shape != (size[0], size[1], 3):
        raise ValueError(f"Unexpected preprocessed shape: {array.shape}")
    return array[None, ...], image


def validate_finite_image(batch: np.ndarray) -> None:
    if not np.isfinite(batch).all():
        raise ValueError("Image contains NaN or infinite pixel values.")
    if float(batch.min()) < 0.0 or float(batch.max()) > 1.0:
        raise ValueError("Preprocessed pixels must be in the [0, 1] range.")
