"""Reusable, lazy-loading inference pipeline used by the Gradio application."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import IMAGE_SIZE, MASK_THRESHOLD, MODEL_METADATA_PATH, MODEL_PATH
from .image_preprocessing import preprocess_image
from .mask_preprocessing import postprocess_probability_map, preprocess_mask
from .metrics import dice_coefficient_np, iou_score_np
from .visualization import create_overlay, mask_to_pil, probability_to_heatmap


class InferenceEngine:
    """Thread-safe lazy model loader for local and Hugging Face Spaces inference."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        metadata_path: str | Path = MODEL_METADATA_PATH,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self._model = model
        self._lock = threading.Lock()
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if not self.model_path.exists():
                    raise FileNotFoundError(
                        f"Model artifact was not found at: {self.model_path}. "
                        "Place unet_medical.keras in the models folder."
                    )
                try:
                    import tensorflow as tf
                except ImportError as exc:
                    raise RuntimeError(
                        "TensorFlow is required for model inference. Install dependencies with "
                        "'pip install -r requirements.txt'."
                    ) from exc
                # compile=False avoids coupling inference to the notebook's custom metric objects.
                self._model = tf.keras.models.load_model(self.model_path, compile=False)
        return self._model

    def predict(
        self,
        image: Any,
        threshold: float = MASK_THRESHOLD,
        ground_truth: Any | None = None,
    ) -> dict[str, Any]:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("Mask threshold must be between 0 and 1.")

        batch, display_image, original_size = preprocess_image(image, IMAGE_SIZE)
        model = self.load_model()
        probability = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        if probability.shape != (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1):
            raise RuntimeError(f"Unexpected model output shape: {probability.shape}")

        restored_probability, restored_mask = postprocess_probability_map(
            probability[0], original_size=original_size, threshold=float(threshold)
        )
        mask_image = mask_to_pil(restored_mask)
        overlay_image = create_overlay(display_image, restored_mask)
        probability_image = probability_to_heatmap(restored_probability)

        metrics: dict[str, Any] = {
            "threshold": round(float(threshold), 3),
            "predicted_region_percent": round(float(restored_mask.mean() * 100.0), 3),
            "mean_probability": round(float(restored_probability.mean()), 5),
            "input_size": f"{display_image.width}×{display_image.height}",
            "model_input": "64×64 grayscale",
            "task": "binary synthetic-region segmentation",
            "clinical_use": "not permitted",
        }

        if ground_truth is not None:
            truth_batch = preprocess_mask(ground_truth, image_size=IMAGE_SIZE)
            hard_model_mask = (probability >= float(threshold)).astype(np.float32)
            metrics["dice_against_uploaded_mask"] = round(
                dice_coefficient_np(truth_batch, hard_model_mask), 6
            )
            metrics["iou_against_uploaded_mask"] = round(
                iou_score_np(truth_batch, hard_model_mask), 6
            )

        with tempfile.NamedTemporaryFile(
            prefix="predicted_mask_", suffix=".png", delete=False
        ) as handle:
            download_path = Path(handle.name)
        mask_image.save(download_path)

        return {
            "original": display_image,
            "mask": mask_image,
            "overlay": overlay_image,
            "probability": probability_image,
            "metrics": metrics,
            "download_path": str(download_path),
        }


def load_segmentation_model(model_path: str | Path = MODEL_PATH) -> Any:
    return InferenceEngine(model_path=model_path).load_model()


def preprocess_uploaded_image(image: Any) -> tuple[np.ndarray, Image.Image, tuple[int, int]]:
    return preprocess_image(image, IMAGE_SIZE)


def predict_mask(
    image: Any,
    threshold: float = MASK_THRESHOLD,
    engine: InferenceEngine | None = None,
) -> dict[str, Any]:
    active_engine = engine or InferenceEngine()
    return active_engine.predict(image=image, threshold=threshold)
