"""Reusable inference functions for Python and the Gradio fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .class_mapping import load_class_mapping
from .config import CLASS_MAPPING_PATH, SIMILAR_CLASS_THRESHOLD, SOURCE_MODEL_PATH
from .image_preprocessing import preprocess_for_source_model


def load_classification_model(path: str | Path = SOURCE_MODEL_PATH):
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required to load the Keras model.") from exc
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def get_top_k_predictions(
    probabilities: Iterable[float],
    class_mapping: dict[int, str],
    k: int = 5,
) -> list[dict[str, float | int | str]]:
    values = np.asarray(list(probabilities), dtype=np.float64).reshape(-1)
    if len(values) != len(class_mapping):
        raise ValueError("Probability count does not match the class mapping.")
    k = max(1, min(int(k), len(values)))
    indices = np.argsort(values)[::-1][:k]
    return [
        {
            "rank": rank,
            "class_index": int(index),
            "class_name": class_mapping[int(index)],
            "probability": float(values[index]),
        }
        for rank, index in enumerate(indices, start=1)
    ]


def detect_close_class_confusion(
    top_predictions: list[dict[str, float | int | str]],
    threshold: float = SIMILAR_CLASS_THRESHOLD,
) -> dict[str, float | bool | str]:
    if len(top_predictions) < 2:
        return {"is_close": False, "probability_gap": 1.0, "message": "Only one class is available."}
    gap = float(top_predictions[0]["probability"]) - float(top_predictions[1]["probability"])
    is_close = gap < threshold
    message = (
        "The top two predictions are close. This may indicate visual similarity between "
        "classes or model uncertainty."
        if is_close
        else "The top prediction is separated from the second prediction by the configured threshold."
    )
    return {"is_close": is_close, "probability_gap": gap, "message": message}


def create_prediction_summary(
    top_predictions: list[dict[str, float | int | str]],
    confusion: dict[str, float | bool | str],
) -> str:
    winner = top_predictions[0]
    statement = (
        f"The VGG16 model predicts '{winner['class_name']}' with "
        f"{float(winner['probability']):.1%} confidence."
    )
    if bool(confusion["is_close"]):
        statement += " The leading classes are visually competitive, so the result should be interpreted cautiously."
    return statement


def predict_class(
    model,
    image,
    *,
    class_mapping: dict[int, str] | None = None,
    top_k: int = 5,
    warning_threshold: float = SIMILAR_CLASS_THRESHOLD,
) -> dict:
    mapping = class_mapping or load_class_mapping(CLASS_MAPPING_PATH)
    batch = preprocess_for_source_model(image)
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    top_predictions = get_top_k_predictions(probabilities, mapping, top_k)
    confusion = detect_close_class_confusion(top_predictions, warning_threshold)
    return {
        "predicted_class": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["probability"],
        "top_predictions": top_predictions,
        "similar_class_warning": confusion,
        "summary": create_prediction_summary(top_predictions, confusion),
    }
