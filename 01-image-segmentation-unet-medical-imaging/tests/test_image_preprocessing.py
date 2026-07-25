import numpy as np
from PIL import Image

from src.image_preprocessing import preprocess_image


def test_preprocess_image_shape_range_and_display_size():
    array = np.linspace(0, 255, 80 * 100, dtype=np.uint8).reshape(80, 100)
    batch, display, original_size = preprocess_image(array)
    assert batch.shape == (1, 64, 64, 1)
    assert batch.dtype == np.float32
    assert float(batch.min()) >= 0.0
    assert float(batch.max()) <= 1.0
    assert display.mode == "RGB"
    assert original_size == (100, 80)


def test_preprocess_rgb_image_converts_to_grayscale_model_input():
    image = Image.new("RGB", (32, 20), color=(120, 60, 10))
    batch, _, _ = preprocess_image(image)
    assert batch.shape == (1, 64, 64, 1)
