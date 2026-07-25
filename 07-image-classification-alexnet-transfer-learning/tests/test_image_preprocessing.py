from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.image_preprocessing import ImagePreprocessingError, preprocess_image_file, preprocess_pil_image


def test_preprocess_pil_image_shape_and_range():
    image = Image.new("RGB", (40, 20), (255, 128, 0))
    array = preprocess_pil_image(image, (227, 227), "zero_one")
    assert array.shape == (1, 227, 227, 3)
    assert array.dtype == np.float32
    assert float(array.min()) >= 0.0
    assert float(array.max()) <= 1.0


def test_bytes_input_is_supported():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    array = preprocess_image_file(buffer.getvalue(), (16, 16))
    assert array.shape == (1, 16, 16, 3)


def test_corrupt_image_is_rejected():
    with pytest.raises(ImagePreprocessingError):
        preprocess_image_file(b"not-an-image")
