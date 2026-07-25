from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.image_preprocessing import (
    ImagePreprocessingError,
    load_rgb_image,
    preprocess_for_browser_model,
    preprocess_for_source_model,
    validate_image_extension,
)


def png_bytes(size=(20, 12), color=(10, 20, 30)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_source_preprocessing_shape_range():
    batch = preprocess_for_source_model(png_bytes())
    assert batch.shape == (1, 32, 32, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0


def test_browser_preprocessing_shape_and_dtype():
    batch = preprocess_for_browser_model(png_bytes())
    assert batch.shape == (1, 96, 96, 3)
    assert batch.dtype == np.float32


def test_corrupt_image_rejected():
    with pytest.raises(ImagePreprocessingError):
        load_rgb_image(b"not an image")


def test_unsupported_extension_rejected():
    with pytest.raises(ImagePreprocessingError):
        validate_image_extension("photo.tiff")
