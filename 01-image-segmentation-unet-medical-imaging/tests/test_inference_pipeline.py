import json

import numpy as np

from src.inference_pipeline import InferenceEngine


class FakeSegmentationModel:
    def predict(self, batch, verbose=0):
        return (batch > 0.55).astype(np.float32)


def test_inference_pipeline_with_injected_model(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"task": "test"}), encoding="utf-8")
    image = np.full((64, 64), 80, dtype=np.uint8)
    image[20:44, 20:44] = 230

    engine = InferenceEngine(
        model_path=tmp_path / "unused.keras",
        metadata_path=metadata_path,
        model=FakeSegmentationModel(),
    )
    result = engine.predict(image, threshold=0.5)

    assert result["original"].size == (64, 64)
    assert result["mask"].size == (64, 64)
    assert result["overlay"].size == (64, 64)
    assert result["probability"].size == (64, 64)
    assert result["metrics"]["predicted_region_percent"] > 0
