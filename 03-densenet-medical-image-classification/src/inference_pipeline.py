"""Reusable, lazy-loading inference pipeline for the bundled Keras artifact."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .class_mapping import humanize_label, load_metadata
from .config import METADATA_PATH, MODEL_PATH
from .image_preprocessing import preprocess_image


class ModelLoadError(RuntimeError):
    """Raised when the Keras model cannot be loaded."""


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    interpretation: str
    warning: str
    gradcam_image: Image.Image | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "interpretation": self.interpretation,
            "warning": self.warning,
        }


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelLoadError(
            "TensorFlow is required for model inference. Install the full requirements.txt file."
        ) from exc
    return tf


@lru_cache(maxsize=2)
def load_classification_model(model_path: str | Path = MODEL_PATH):
    tf = _import_tensorflow()
    path = Path(model_path)
    if not path.exists():
        raise ModelLoadError(f"Trained model was not found: {path}")
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise ModelLoadError(f"Unable to load Keras model '{path.name}': {exc}") from exc


def _normalize_probabilities(raw_output: np.ndarray, class_count: int) -> np.ndarray:
    values = np.asarray(raw_output, dtype=np.float64).reshape(-1)
    if values.size == 1 and class_count == 2:
        positive = float(np.clip(values[0], 0.0, 1.0))
        values = np.array([1.0 - positive, positive], dtype=np.float64)
    if values.size != class_count:
        raise ValueError(
            f"Model returned {values.size} values, but metadata defines {class_count} classes."
        )
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-3):
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        values = exp_values / exp_values.sum()
    return values / values.sum()


def generate_gradcam_heatmap(
    model: Any,
    batch: np.ndarray,
    display_image: Image.Image,
    class_index: int,
) -> Image.Image | None:
    """Generate a coarse Grad-CAM overlay from the nested DenseNet output.

    The function returns None when the installed backend/model graph does not expose
    a suitable feature-map layer. The app treats explainability as optional.
    """
    try:
        tf = _import_tensorflow()
        candidate = None
        for layer in reversed(model.layers):
            shape = getattr(layer, "output_shape", None)
            if shape is None:
                try:
                    shape = tuple(layer.output.shape)
                except Exception:
                    continue
            if isinstance(shape, list):
                continue
            if len(shape) == 4:
                candidate = layer
                break
        if candidate is None:
            return None
        grad_model = tf.keras.Model(model.inputs, [candidate.output, model.output])
        with tf.GradientTape() as tape:
            feature_maps, predictions = grad_model(batch, training=False)
            score = predictions[:, class_index]
        gradients = tape.gradient(score, feature_maps)
        if gradients is None:
            return None
        pooled = tf.reduce_mean(gradients, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(feature_maps[0] * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        denominator = tf.reduce_max(heatmap)
        if float(denominator.numpy()) <= 0:
            return None
        heatmap = (heatmap / denominator).numpy()
        heat = Image.fromarray(np.uint8(heatmap * 255), mode="L")
        heat = heat.resize(display_image.size, Image.Resampling.BILINEAR)
        # Use a simple red overlay without adding another plotting dependency.
        rgba = Image.new("RGBA", display_image.size, (255, 0, 0, 0))
        rgba.putalpha(heat.point(lambda value: int(value * 0.55)))
        return Image.alpha_composite(display_image.convert("RGBA"), rgba).convert("RGB")
    except Exception:
        return None


def predict_class(
    image: Any,
    model: Any | None = None,
    metadata_path: str | Path = METADATA_PATH,
    include_gradcam: bool = True,
) -> PredictionResult:
    metadata = load_metadata(metadata_path)
    classes = metadata["classes"]
    batch, display_image = preprocess_image(image, metadata["input_shape"])
    active_model = model if model is not None else load_classification_model()
    raw = active_model.predict(batch, verbose=0)
    probabilities = _normalize_probabilities(raw[0], len(classes))
    index = int(np.argmax(probabilities))
    predicted = classes[index]
    confidence = float(probabilities[index])
    probability_map = {
        label: float(probabilities[position]) for position, label in enumerate(classes)
    }
    warning = (
        "The bundled model is a synthetic Fashion-MNIST proxy and was not trained on clinical chest X-rays. "
        "Do not interpret this output as pneumonia detection."
        if metadata.get("dataset_status") == "synthetic_proxy_not_clinical"
        else "Educational model output only; not medical advice."
    )
    interpretation = (
        f"The model assigned the highest probability to '{humanize_label(predicted)}' "
        f"with {confidence:.1%} confidence. {warning}"
    )
    gradcam = generate_gradcam_heatmap(active_model, batch, display_image, index) if include_gradcam else None
    return PredictionResult(
        predicted_class=predicted,
        confidence=confidence,
        probabilities=probability_map,
        interpretation=interpretation,
        warning=warning,
        gradcam_image=gradcam,
    )


def get_top_predictions(result: PredictionResult, top_k: int = 3) -> list[dict[str, float | str]]:
    ranked = sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)
    return [
        {"class": label, "probability": probability, "percentage": probability * 100.0}
        for label, probability in ranked[: max(1, top_k)]
    ]


def create_prediction_summary(result: PredictionResult) -> str:
    payload = result.to_dict()
    payload["top_predictions"] = get_top_predictions(result)
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="densenet_prediction_", delete=False, encoding="utf-8"
    )
    with handle:
        json.dump(payload, handle, indent=2)
    return handle.name
