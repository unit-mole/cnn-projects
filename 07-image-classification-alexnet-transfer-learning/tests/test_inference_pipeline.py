import numpy as np

from src.inference_pipeline import create_prediction_summary, get_top_k_predictions


def test_top_k_predictions_are_sorted():
    result = get_top_k_predictions(np.array([0.1, 0.7, 0.2]), ["a", "b", "c"], 2)
    assert [row["class_name"] for row in result] == ["b", "c"]
    assert result[0]["confidence"] == 0.7


def test_prediction_summary_contains_best_class():
    summary = create_prediction_summary([{"class_name": "frog", "confidence": 0.82}])
    assert "frog" in summary
    assert "82.0%" in summary
