"""Image loading and preprocessing for both training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import ALLOWED_IMAGE_EXTENSIONS, IMAGE_SIZE


def _array_to_pil(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image array shape: {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(finite)) <= 1.0:
            finite = finite * 255.0
        arr = np.clip(finite, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def load_image(image: Any) -> Image.Image:
    """Load a path, PIL image, or NumPy array and return an EXIF-corrected PIL image."""
    if image is None:
        raise ValueError("No image was provided.")
    if isinstance(image, Image.Image):
        loaded = image.copy()
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{path.suffix}'. "
                f"Allowed extensions: {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
            )
        try:
            loaded = Image.open(path)
        except Exception as exc:  # pragma: no cover - PIL supplies detailed error
            raise ValueError(f"The image could not be opened: {exc}") from exc
    elif isinstance(image, np.ndarray):
        loaded = _array_to_pil(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")
    return ImageOps.exif_transpose(loaded)


def preprocess_image(
    image: Any,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, Image.Image, tuple[int, int]]:
    """Convert input to grayscale, resize with bilinear interpolation, and normalize.

    Returns
    -------
    batch:
        Float32 array with shape ``(1, height, width, 1)`` and values in ``[0, 1]``.
    display_image:
        RGB copy at the original input resolution for visualization.
    original_size:
        ``(width, height)`` used to restore model outputs to display resolution.
    """
    pil_image = load_image(image)
    if pil_image.width < 2 or pil_image.height < 2:
        raise ValueError("The image is too small. Upload an image at least 2×2 pixels.")

    original_size = pil_image.size
    display_image = pil_image.convert("RGB")
    grayscale = pil_image.convert("L")
    resized = grayscale.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = array[None, ..., None]
    return batch, display_image, original_size
