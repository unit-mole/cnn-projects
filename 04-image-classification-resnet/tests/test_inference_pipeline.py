import numpy as np
from PIL import Image

from src.inference_pipeline import (
    create_prediction_summary,
    get_top_predictions,
    preprocess_for_loaded_model,
)


class FakeModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape


def test_top_predictions_are_sorted():
    probabilities = np.linspace(0.0, 1.0, 100)
    names = tuple(f"class_{i}" for i in range(100))
    top = get_top_predictions(probabilities, names, k=3)
    assert [name for name, _ in top] == ["class_99", "class_98", "class_97"]
    assert top[0][1] >= top[1][1] >= top[2][1]


def test_prediction_summary_contains_label_and_confidence():
    summary = create_prediction_summary("apple", 0.8)
    assert "apple" in summary
    assert "80.0%" in summary


def test_training_model_preprocessing_is_normalized_32_rgb():
    image = Image.new("RGB", (50, 40), (255, 0, 0))
    batch = preprocess_for_loaded_model(FakeModel((None, 32, 32, 3)), image)
    assert batch.shape == (1, 32, 32, 3)
    np.testing.assert_allclose(batch[0, 0, 0], [1.0, 0.0, 0.0], atol=1e-6)


def test_browser_model_preprocessing_is_96_bgr_mean_subtracted():
    image = Image.new("RGB", (50, 40), (255, 0, 0))
    batch = preprocess_for_loaded_model(FakeModel((None, 96, 96, 3)), image)
    assert batch.shape == (1, 96, 96, 3)
    np.testing.assert_allclose(
        batch[0, 0, 0],
        [-103.939, -116.779, 131.32],
        atol=1e-3,
    )
