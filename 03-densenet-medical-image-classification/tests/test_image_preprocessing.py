import numpy as np
from PIL import Image

from src.image_preprocessing import ImagePreprocessingError, preprocess_image


def test_preprocess_rgb_shape_and_range():
    image = Image.new("L", (64, 32), color=128)
    batch, display = preprocess_image(image, [28, 28, 3])
    assert batch.shape == (1, 28, 28, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0
    assert display.mode == "RGB"


def test_invalid_array_rank_raises():
    invalid = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    try:
        preprocess_image(invalid, [28, 28, 3])
    except ImagePreprocessingError:
        return
    raise AssertionError("Expected ImagePreprocessingError")
