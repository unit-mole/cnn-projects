from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from .class_mapping import CIFAR100_FINE_LABELS
from .image_preprocessing import preprocess_image, validate_preprocessed_batch


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    confidence: float
    top_predictions: tuple[tuple[str, float], ...]
    summary: str


def load_classification_model(path: str | Path) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required to load the Keras model.") from exc
    # This project loads its own trusted artifact. Do not disable safe mode for unknown models.
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def _open_rgb(source: Any) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            return source.copy().convert("RGB")
        if isinstance(source, (bytes, bytearray)):
            return Image.open(BytesIO(source)).convert("RGB")
        return Image.open(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The supplied file is not a readable image.") from exc


def preprocess_for_loaded_model(model: Any, image_source: Any) -> np.ndarray:
    """Select preprocessing from the loaded model's declared input shape.

    The supplied training model accepts normalized 32×32 RGB data and contains
    resize/ResNet preprocessing internally. The browser-export H5 model accepts
    an externally preprocessed 96×96 BGR tensor.
    """
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) != 4:
        raise ValueError(f"Unsupported model input shape: {shape}")
    height, width, channels = shape[1], shape[2], shape[3]
    if channels != 3:
        raise ValueError(f"Expected a three-channel image model, received input shape {shape}.")

    if (height, width) == (32, 32):
        image = _open_rgb(image_source).resize((32, 32), Image.Resampling.BILINEAR)
        return np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)
    if (height, width) == (96, 96):
        batch = preprocess_image(image_source, (96, 96))
        validate_preprocessed_batch(batch)
        return batch
    raise ValueError(
        f"Unsupported model input size {(height, width)}. Expected 32×32 training model or 96×96 browser model."
    )


def get_top_predictions(
    probabilities: np.ndarray,
    class_names: tuple[str, ...] = CIFAR100_FINE_LABELS,
    k: int = 3,
) -> tuple[tuple[str, float], ...]:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(values) != len(class_names):
        raise ValueError("Probability vector length does not match class mapping.")
    if not 1 <= k <= len(class_names):
        raise ValueError("k must be between 1 and the number of classes.")
    indices = np.argsort(values)[::-1][:k]
    return tuple((class_names[int(i)], float(values[i])) for i in indices)


def create_prediction_summary(predicted_class: str, confidence: float) -> str:
    qualifier = "high" if confidence >= 0.75 else "moderate" if confidence >= 0.45 else "low"
    return (
        f"The model's highest-scoring class is '{predicted_class}' with {confidence:.1%} "
        f"confidence. Treat this as a {qualifier}-confidence model estimate, not guaranteed truth."
    )


def predict_class(
    model: Any,
    image_source: Any,
    class_names: tuple[str, ...] = CIFAR100_FINE_LABELS,
    top_k: int = 3,
) -> Prediction:
    batch = preprocess_for_loaded_model(model, image_source)
    output = model.predict(batch, verbose=0)
    probabilities = np.asarray(output, dtype=np.float64).reshape(-1)
    top = get_top_predictions(probabilities, class_names, top_k)
    predicted_class, confidence = top[0]
    return Prediction(
        predicted_class=predicted_class,
        confidence=confidence,
        top_predictions=top,
        summary=create_prediction_summary(predicted_class, confidence),
    )
