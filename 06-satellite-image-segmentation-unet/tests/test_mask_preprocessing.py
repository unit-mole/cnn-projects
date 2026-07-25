import numpy as np
from PIL import Image

from src.mask_preprocessing import preprocess_binary_mask


def test_binary_mask_uses_nearest_and_binary_values():
    array = np.zeros((8, 8), dtype=np.uint8)
    array[2:6, 2:6] = 255
    mask = preprocess_binary_mask(Image.fromarray(array), (64, 64))
    assert mask.shape == (64, 64)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
