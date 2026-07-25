import numpy as np

from src.mask_preprocessing import postprocess_probability_map, preprocess_mask


def test_preprocess_mask_is_binary_and_uses_expected_shape():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 8:22] = 255
    processed = preprocess_mask(mask)
    assert processed.shape == (1, 64, 64, 1)
    assert set(np.unique(processed)).issubset({0.0, 1.0})


def test_postprocess_probability_map_restores_original_size():
    probability = np.zeros((64, 64, 1), dtype=np.float32)
    probability[16:48, 16:48, 0] = 0.9
    restored_probability, restored_mask = postprocess_probability_map(
        probability, original_size=(120, 80), threshold=0.5
    )
    assert restored_probability.shape == (80, 120)
    assert restored_mask.shape == (80, 120)
    assert set(np.unique(restored_mask)).issubset({0, 1})
