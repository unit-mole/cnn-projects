"""Higher-level classification API used by scripts and external callers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .inference_pipeline import PredictionResult, predict_class


def classify_medical_image(
    image: Any,
    *,
    metadata_path: str | Path | None = None,
    include_gradcam: bool = True,
) -> PredictionResult:
    kwargs = {"include_gradcam": include_gradcam}
    if metadata_path is not None:
        kwargs["metadata_path"] = metadata_path
    return predict_class(image, **kwargs)
