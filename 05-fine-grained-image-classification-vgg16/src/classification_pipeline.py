"""High-level end-to-end classification entry point."""

from __future__ import annotations

from pathlib import Path

from .config import SOURCE_MODEL_PATH
from .inference_pipeline import load_classification_model, predict_class


def classify_image(image_path: str | Path, model_path: str | Path = SOURCE_MODEL_PATH) -> dict:
    model = load_classification_model(model_path)
    return predict_class(model, image_path)
