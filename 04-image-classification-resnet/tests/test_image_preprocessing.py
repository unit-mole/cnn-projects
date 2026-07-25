import numpy as np
from PIL import Image

from src.image_preprocessing import preprocess_image, validate_preprocessed_batch


def test_preprocess_image_shape_and_dtype():
    image = Image.new("RGB", (32, 32), (255, 0, 0))
    batch = preprocess_image(image)
    assert batch.shape == (1, 96, 96, 3)
    assert batch.dtype == np.float32
    validate_preprocessed_batch(batch)


def test_preprocess_matches_resnet_bgr_channel_order():
    image = Image.new("RGB", (1, 1), (255, 0, 0))
    pixel = preprocess_image(image, (1, 1))[0, 0, 0]
    expected = np.array([0.0 - 103.939, 0.0 - 116.779, 255.0 - 123.68], dtype=np.float32)
    np.testing.assert_allclose(pixel, expected, atol=1e-4)
