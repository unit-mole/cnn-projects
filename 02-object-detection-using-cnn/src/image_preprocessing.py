from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy()
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        if array.ndim == 3 and array.shape[2] in (3, 4):
            return Image.fromarray(array)
    raise TypeError("Expected a file path, PIL image, or NumPy image array.")


def preprocess_uploaded_image(
    image: ImageInput,
    image_size: tuple[int, int] = (64, 64),
    auto_invert: bool = True,
) -> tuple[np.ndarray, Image.Image]:
    """Convert an uploaded image to the model's 64x64 grayscale input.

    The model was trained on light MNIST digits over a dark canvas. When the
    uploaded image has a bright background, optional automatic inversion makes
    it closer to the training distribution.
    """
    pil = _to_pil(image).convert("L")
    pil = ImageOps.exif_transpose(pil)

    if auto_invert and float(np.asarray(pil).mean()) > 127.0:
        pil = ImageOps.invert(pil)

    resized = pil.resize(image_size, Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = array[np.newaxis, ..., np.newaxis]
    return batch, resized.convert("RGB")


def validate_model_input(batch: np.ndarray) -> None:
    if batch.shape != (1, 64, 64, 1):
        raise ValueError(f"Expected model input shape (1, 64, 64, 1), got {batch.shape}.")
    if batch.dtype != np.float32:
        raise TypeError(f"Expected float32 input, got {batch.dtype}.")
    if np.min(batch) < 0.0 or np.max(batch) > 1.0:
        raise ValueError("Model input must be normalized to [0, 1].")
