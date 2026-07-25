from pathlib import Path

import numpy as np
from PIL import Image

from src.inference_pipeline import load_detection_model, predict_objects


def test_model_loads():
    assert load_detection_model() is not None


def test_prediction_contract():
    image = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
    annotated, table, details = predict_objects(image, confidence_threshold=0.0)
    assert annotated.size == (512, 512)
    assert len(table) == 1
    assert "predicted_digit" in details
    assert len(details["normalized_box_xyxy"]) == 4
