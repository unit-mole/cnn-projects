import numpy as np

from src.inference_pipeline import (
    create_prediction_summary,
    detect_close_class_confusion,
    get_top_k_predictions,
)


def test_top_predictions_are_sorted():
    result = get_top_k_predictions([0.2, 0.8], {0: "cat", 1: "dog"}, k=5)
    assert [row["class_name"] for row in result] == ["dog", "cat"]
    assert result[0]["probability"] == 0.8


def test_close_prediction_warning():
    top = get_top_k_predictions([0.46, 0.54], {0: "cat", 1: "dog"})
    warning = detect_close_class_confusion(top, threshold=0.15)
    assert warning["is_close"] is True
    assert np.isclose(warning["probability_gap"], 0.08)
    assert "close" in create_prediction_summary(top, warning).lower() or "cautiously" in create_prediction_summary(top, warning).lower()


def test_confident_prediction_has_no_warning():
    top = get_top_k_predictions([0.05, 0.95], {0: "cat", 1: "dog"})
    warning = detect_close_class_confusion(top, threshold=0.15)
    assert warning["is_close"] is False
