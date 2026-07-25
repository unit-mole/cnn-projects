from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGENET_BGR_MEANS = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)


def _open_rgb(source: str | Path | bytes | bytearray | BinaryIO | Image.Image) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(BytesIO(source))
        else:
            image = Image.open(source)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The supplied file is not a readable image.") from exc


def preprocess_image(
    source: str | Path | bytes | bytearray | BinaryIO | Image.Image,
    image_size: tuple[int, int] = (96, 96),
) -> np.ndarray:
    """Return a batched ResNet50 input matching keras.applications preprocessing.

    Input is decoded as RGB, resized, converted to float32, reordered to BGR,
    and ImageNet channel means are subtracted.
    """
    image = _open_rgb(source)
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32)
    bgr = rgb[..., ::-1]
    bgr -= IMAGENET_BGR_MEANS
    return np.expand_dims(bgr, axis=0)


def validate_preprocessed_batch(batch: np.ndarray) -> None:
    if batch.dtype != np.float32:
        raise TypeError(f"Expected float32 input, received {batch.dtype}.")
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB/BGR batch, received shape {batch.shape}.")
    if not np.isfinite(batch).all():
        raise ValueError("Preprocessed image contains NaN or infinite values.")
