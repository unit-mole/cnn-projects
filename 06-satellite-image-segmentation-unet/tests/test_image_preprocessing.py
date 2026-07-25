import numpy as np
from PIL import Image

from src.image_preprocessing import preprocess_image


def test_preprocess_image_shape_and_range():
    image = Image.fromarray(np.full((100, 120, 3), 128, dtype=np.uint8))
    batch, original = preprocess_image(image, (64, 64))
    assert batch.shape == (1, 64, 64, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0
    assert original.size == (120, 100)
