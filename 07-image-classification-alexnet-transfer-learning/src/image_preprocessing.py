from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
Normalization = Literal["zero_one", "minus_one_one", "none"]


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be safely decoded or prepared."""


def _normalize(array: np.ndarray, method: Normalization) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if method == "zero_one":
        return array / 255.0
    if method == "minus_one_one":
        return array / 127.5 - 1.0
    if method == "none":
        return array
    raise ValueError(f"Unsupported normalization method: {method}")


def preprocess_pil_image(
    image: Image.Image,
    image_size: tuple[int, int] = (227, 227),
    normalization: Normalization = "zero_one",
    add_batch_dimension: bool = True,
) -> np.ndarray:
    if image.width <= 0 or image.height <= 0:
        raise ImagePreprocessingError("Image dimensions must be positive.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    array = _normalize(np.asarray(image), normalization)
    if array.shape != (image_size[1], image_size[0], 3):
        raise ImagePreprocessingError(f"Unexpected processed image shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ImagePreprocessingError("Processed image contains non-finite values.")
    return np.expand_dims(array, axis=0) if add_batch_dimension else array


def load_image(source: str | Path | bytes | BinaryIO) -> Image.Image:
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ImagePreprocessingError(
                    f"Unsupported image extension '{path.suffix}'. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )
            with Image.open(path) as image:
                image.load()
                return image.copy()
        if isinstance(source, bytes):
            with Image.open(BytesIO(source)) as image:
                image.load()
                return image.copy()
        with Image.open(source) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImagePreprocessingError("The image is corrupt or cannot be decoded.") from exc


def preprocess_image_file(
    source: str | Path | bytes | BinaryIO,
    image_size: tuple[int, int] = (227, 227),
    normalization: Normalization = "zero_one",
) -> np.ndarray:
    return preprocess_pil_image(load_image(source), image_size, normalization, True)
