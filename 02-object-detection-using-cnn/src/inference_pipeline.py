from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from .bounding_box_utils import sanitize_normalized_xyxy
from .config import METADATA_PATH, MODEL_PATH
from .image_preprocessing import preprocess_uploaded_image, validate_model_input
from .visualization import draw_detection


@lru_cache(maxsize=1)
def load_detection_model(model_path: str | Path = MODEL_PATH) -> tf.keras.Model:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return tf.keras.models.load_model(path, compile=False)


@lru_cache(maxsize=1)
def load_model_metadata(metadata_path: str | Path = METADATA_PATH) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_objects(
    image: Image.Image | np.ndarray | str | Path,
    confidence_threshold: float = 0.50,
    auto_invert: bool = True,
) -> tuple[Image.Image, pd.DataFrame, dict[str, Any]]:
    """Run the single-object detector and return image, table, and details."""
    model = load_detection_model()
    metadata = load_model_metadata()

    batch, display_image = preprocess_uploaded_image(image, auto_invert=auto_invert)
    validate_model_input(batch)

    class_probabilities, raw_box = model.predict(batch, verbose=0)
    probabilities = np.asarray(class_probabilities[0], dtype=np.float32)
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    box = sanitize_normalized_xyxy(raw_box[0])

    passed = confidence >= float(confidence_threshold)
    label = metadata["class_names"][predicted_class]

    if passed:
        annotated = draw_detection(display_image.resize((512, 512)), box.tolist(), label, confidence)
        rows = [{
            "detected_class": label,
            "confidence": round(confidence, 4),
            "x1_normalized": round(float(box[0]), 4),
            "y1_normalized": round(float(box[1]), 4),
            "x2_normalized": round(float(box[2]), 4),
            "y2_normalized": round(float(box[3]), 4),
        }]
    else:
        annotated = display_image.resize((512, 512))
        rows = []

    top3_indices = np.argsort(probabilities)[::-1][:3]
    details = {
        "status": "Detection shown" if passed else "Below confidence threshold",
        "predicted_digit": label,
        "confidence": round(confidence, 4),
        "confidence_threshold": round(float(confidence_threshold), 4),
        "normalized_box_xyxy": [round(float(v), 4) for v in box],
        "top_3_predictions": [
            {"digit": metadata["class_names"][int(i)], "probability": round(float(probabilities[i]), 4)}
            for i in top3_indices
        ],
        "model_scope": "One handwritten digit per 64x64 grayscale image",
        "nms_applied": False,
    }
    return annotated, pd.DataFrame(rows), details
