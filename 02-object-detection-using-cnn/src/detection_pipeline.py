"""Public detection API used by the Gradio application."""

from .inference_pipeline import (
    load_detection_model,
    load_model_metadata,
    predict_objects,
)

__all__ = ["load_detection_model", "load_model_metadata", "predict_objects"]
