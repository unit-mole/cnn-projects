import numpy as np
from PIL import Image

from src.config import InferenceConfig
from src.inference_pipeline import InferencePipeline


class DummyModel:
    def predict(self, batch, verbose=0):
        probabilities = batch.mean(axis=-1, keepdims=True)
        return probabilities.astype(np.float32)


def test_pipeline_with_injected_model():
    image = Image.fromarray(np.full((32, 40, 3), 200, dtype=np.uint8))
    pipeline = InferencePipeline(
        config=InferenceConfig(height=64, width=64, channels=3, threshold=0.5),
        model=DummyModel(),
    )
    result = pipeline.predict(image)
    assert result.probability.shape == (64, 64)
    assert result.binary_mask.shape == (64, 64)
    assert result.mask_image.size == image.size
    assert result.overlay_image.size == image.size
