import numpy as np

from src.synthetic_data import generate_synthetic_dataset, make_medical_sample, split_dataset


def test_synthetic_sample_is_deterministic_and_aligned():
    image_a, mask_a = make_medical_sample(seed=42)
    image_b, mask_b = make_medical_sample(seed=42)
    assert image_a.shape == (64, 64, 1)
    assert mask_a.shape == (64, 64, 1)
    assert np.array_equal(image_a, image_b)
    assert np.array_equal(mask_a, mask_b)
    assert float(image_a[mask_a.astype(bool)].mean()) > float(image_a[~mask_a.astype(bool)].mean())


def test_split_sizes_match_original_notebook_ratio():
    images, masks = generate_synthetic_dataset(num_samples=100, seed=42)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(images, masks, seed=42)
    assert len(x_train) == 70
    assert len(x_val) == 15
    assert len(x_test) == 15
    assert len(y_train) == 70
    assert len(y_val) == 15
    assert len(y_test) == 15
