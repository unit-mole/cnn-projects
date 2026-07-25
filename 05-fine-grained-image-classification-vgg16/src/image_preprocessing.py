"""Image validation and preprocessing shared by training and inference."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MODEL_IMAGE_SIZE, SOURCE_IMAGE_SIZE

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VGG16_BGR_MEANS = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)


class ImagePreprocessingError(ValueError):
    """Raised when an uploaded image cannot be safely decoded or processed."""


def validate_image_extension(filename: str | Path) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ImagePreprocessingError(f"Unsupported image format '{suffix}'. Allowed: {allowed}")


def load_rgb_image(source: str | Path | bytes | bytearray | BinaryIO | Image.Image) -> Image.Image:
    """Decode an image, apply EXIF orientation, and return an independent RGB image."""
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(BytesIO(source))
        else:
            if isinstance(source, (str, Path)):
                validate_image_extension(source)
            image = Image.open(source)
        image.load()
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagePreprocessingError("The image is missing, corrupt, or unsupported.") from exc


def resize_rgb_array(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize with bilinear interpolation and return a float32 RGB array."""
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def preprocess_for_source_model(
    source: str | Path | bytes | bytearray | BinaryIO | Image.Image,
) -> np.ndarray:
    """Prepare input for the original .keras model.

    The source model expects normalized 32x32 RGB input, then performs its own
    96x96 resize and VGG16 preprocessing inside the graph.
    """
    image = load_rgb_image(source)
    array = resize_rgb_array(image, SOURCE_IMAGE_SIZE) / 255.0
    return np.expand_dims(array.astype(np.float32), axis=0)


def preprocess_for_browser_model(
    source: str | Path | bytes | bytearray | BinaryIO | Image.Image,
) -> np.ndarray:
    """Prepare input for the flattened browser model.

    This reproduces the browser pipeline: RGB -> 32x32 -> 96x96 -> BGR and
    ImageNet mean subtraction. No 1/255 scaling is applied after decoding.
    """
    image = load_rgb_image(source)
    source_array = resize_rgb_array(image, SOURCE_IMAGE_SIZE)
    source_image = Image.fromarray(np.clip(source_array, 0, 255).astype(np.uint8), mode="RGB")
    array = resize_rgb_array(source_image, MODEL_IMAGE_SIZE)
    bgr = array[..., ::-1]
    return np.expand_dims((bgr - VGG16_BGR_MEANS).astype(np.float32), axis=0)
