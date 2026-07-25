import numpy as np
from PIL import Image

from src.image_preprocessing import preprocess_uploaded_image, validate_model_input


def test_preprocessing_shape_dtype_and_range():
    image = Image.fromarray(np.full((100, 80), 255, dtype=np.uint8))
    batch, display = preprocess_uploaded_image(image)
    assert batch.shape == (1, 64, 64, 1)
    assert batch.dtype == np.float32
    assert display.size == (64, 64)
    validate_model_input(batch)
