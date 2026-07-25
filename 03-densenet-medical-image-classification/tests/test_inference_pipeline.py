import json

import numpy as np
from PIL import Image

from src.inference_pipeline import get_top_predictions, predict_class


class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 28, 28, 3)
        return np.array([[0.2, 0.8]], dtype=np.float32)


def test_inference_with_injected_model(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({
            "classes": ["normal_like", "pneumonia_like"],
            "class_to_index": {"normal_like": 0, "pneumonia_like": 1},
            "input_shape": [28, 28, 3],
            "dataset_status": "synthetic_proxy_not_clinical",
        }),
        encoding="utf-8",
    )
    image = Image.new("RGB", (40, 40), color="white")
    result = predict_class(
        image,
        model=FakeModel(),
        metadata_path=metadata_path,
        include_gradcam=False,
    )
    assert result.predicted_class == "pneumonia_like"
    assert abs(result.confidence - 0.8) < 1e-6
    assert get_top_predictions(result)[0]["class"] == "pneumonia_like"
