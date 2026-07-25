from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig
from .inference_pipeline import load_classification_model, predict_class
from .model_training import train


def train_and_save(config: ProjectConfig | None = None):
    return train(config)


def classify_image(image_path: str | Path, model_path: str | Path | None = None):
    cfg = ProjectConfig()
    model = load_classification_model(model_path or cfg.keras_model_path)
    return predict_class(model, image_path)
