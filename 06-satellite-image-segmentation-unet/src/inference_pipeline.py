from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .config import MODEL_PATH, InferenceConfig, load_inference_config
from .image_preprocessing import preprocess_image, validate_finite_image
from .visualization import binary_mask_to_image, create_overlay, probability_to_image


class Predictor(Protocol):
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray: ...


@dataclass
class SegmentationResult:
    original: Image.Image
    probability: np.ndarray
    binary_mask: np.ndarray
    mask_image: Image.Image
    overlay_image: Image.Image
    probability_image: Image.Image


class InferencePipeline:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        config: InferenceConfig | None = None,
        model: Predictor | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = config or load_inference_config()
        self._model = model

    def load_model(self) -> Predictor:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Trained model not found: {self.model_path}")
            import tensorflow as tf
            # compile=False avoids deserializing training-only custom metric functions.
            self._model = tf.keras.models.load_model(self.model_path, compile=False)
        return self._model

    def predict(self, source: Any) -> SegmentationResult:
        batch, original = preprocess_image(source, (self.config.height, self.config.width))
        validate_finite_image(batch)
        probabilities = np.asarray(self.load_model().predict(batch, verbose=0), dtype=np.float32)
        expected = (1, self.config.height, self.config.width, 1)
        if probabilities.shape != expected:
            raise ValueError(f"Unexpected model output shape: {probabilities.shape}; expected {expected}")
        probability = np.clip(probabilities[0, ..., 0], 0.0, 1.0)
        binary = (probability >= self.config.threshold).astype(np.uint8)
        return SegmentationResult(
            original=original,
            probability=probability,
            binary_mask=binary,
            mask_image=binary_mask_to_image(binary, original.size),
            overlay_image=create_overlay(original, binary, self.config.overlay_alpha),
            probability_image=probability_to_image(probability, original.size),
        )
