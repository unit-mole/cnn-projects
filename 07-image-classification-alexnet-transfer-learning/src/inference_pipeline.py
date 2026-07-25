from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .image_preprocessing import preprocess_image_file


def load_metadata(path: str | Path) -> dict[str, Any]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"class_names", "input_height", "input_width", "normalization"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Metadata is missing required fields: {sorted(missing)}")
    return metadata


def load_classification_model(model_path: str | Path):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def get_top_k_predictions(
    probabilities: np.ndarray,
    class_names: list[str],
    k: int = 5,
) -> list[dict[str, float | str]]:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(scores) != len(class_names):
        raise ValueError("Probability count does not match class names.")
    k = max(1, min(int(k), len(class_names)))
    order = np.argsort(scores)[::-1][:k]
    return [
        {"class_name": class_names[int(index)], "confidence": float(scores[int(index)])}
        for index in order
    ]


def create_prediction_summary(top_predictions: list[dict]) -> str:
    if not top_predictions:
        raise ValueError("At least one prediction is required.")
    best = top_predictions[0]
    return (
        f"Predicted class: {best['class_name']} with "
        f"{float(best['confidence']) * 100:.1f}% confidence. "
        "Treat this as a model estimate, not guaranteed truth."
    )


def predict_class(
    model,
    image_source,
    metadata: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    batch = preprocess_image_file(
        image_source,
        image_size=(int(metadata["input_width"]), int(metadata["input_height"])),
        normalization=metadata.get("normalization", "zero_one"),
    )
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    top = get_top_k_predictions(probabilities, list(metadata["class_names"]), top_k)
    return {
        "predicted_class": top[0]["class_name"],
        "confidence": top[0]["confidence"],
        "top_predictions": top,
        "summary": create_prediction_summary(top),
    }
