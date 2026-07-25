"""Consistent image preprocessing for local, test, and Gradio inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import SUPPORTED_IMAGE_SUFFIXES


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be safely prepared for inference."""


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy()
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ImagePreprocessingError(
                f"Unsupported image type '{path.suffix}'. Supported: {sorted(SUPPORTED_IMAGE_SUFFIXES)}"
            )
        try:
            with Image.open(path) as loaded:
                return loaded.copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise ImagePreprocessingError(f"The image could not be opened: {exc}") from exc
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise ImagePreprocessingError("NumPy image arrays must have 2 or 3 dimensions.")
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 0.0
            if max_value <= 1.0:
                array = array * 255.0
        array = np.nan_to_num(array).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)
    raise ImagePreprocessingError(
        "Image must be a PIL image, NumPy array, or a supported image-file path."
    )


def preprocess_image(
    image: Any,
    input_shape: tuple[int, int, int] | list[int],
    scale_to_unit_interval: bool = True,
) -> tuple[np.ndarray, Image.Image]:
    """Return a batched float32 tensor and a display-safe RGB image.

    The bundled artifact expects 28x28 RGB pixels scaled to [0, 1]. Its saved
    graph performs the later resize to 96x96 and DenseNet preprocessing.
    """
    height, width, channels = [int(value) for value in input_shape]
    if channels not in (1, 3):
        raise ImagePreprocessingError(f"Only 1- or 3-channel inputs are supported, received {channels}.")

    pil_image = ImageOps.exif_transpose(_to_pil(image))
    display_image = pil_image.convert("RGB")
    model_image = display_image.convert("L" if channels == 1 else "RGB")
    model_image = model_image.resize((width, height), Image.Resampling.BILINEAR)

    array = np.asarray(model_image, dtype=np.float32)
    if channels == 1:
        array = np.expand_dims(array, axis=-1)
    if scale_to_unit_interval:
        array /= 255.0
    array = np.expand_dims(array, axis=0)
    return array, display_image
