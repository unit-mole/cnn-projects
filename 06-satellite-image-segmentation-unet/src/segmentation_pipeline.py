from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .config import METRICS_PATH, MODEL_METADATA_PATH, RUNTIME_DIR, load_json
from .inference_pipeline import InferencePipeline
from .mask_preprocessing import preprocess_binary_mask
from .metrics import dice_coefficient, iou_score, precision_recall_f1


class SegmenterService:
    def __init__(self, pipeline: InferencePipeline | None = None) -> None:
        self.pipeline = pipeline or InferencePipeline()
        self.metadata = load_json(MODEL_METADATA_PATH)
        self.reported_metrics = load_json(METRICS_PATH)

    def segment(self, image: Any, ground_truth: Any | None = None):
        result = self.pipeline.predict(image)
        metrics = {
            "model_scope": "Synthetic 64×64 binary urban-structure benchmark",
            "threshold": self.pipeline.config.threshold,
            "reported_test_metrics": self.reported_metrics,
            "warning": "Near-perfect synthetic scores do not indicate real satellite-image generalization.",
        }
        if ground_truth is not None:
            true_mask = preprocess_binary_mask(
                ground_truth,
                (self.pipeline.config.height, self.pipeline.config.width),
            )
            pred = result.binary_mask.astype("float32")
            metrics["uploaded_ground_truth_metrics"] = {
                "dice": dice_coefficient(true_mask, pred),
                "iou": iou_score(true_mask, pred),
                **precision_recall_f1(true_mask, pred),
            }

        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RUNTIME_DIR / f"predicted_mask_{uuid.uuid4().hex[:10]}.png"
        result.mask_image.save(output_path)
        return result, metrics, output_path

    def model_details(self) -> dict:
        return {
            "task": self.metadata["task"],
            "target": self.metadata["target_region"],
            "input": self.metadata["input"],
            "output": self.metadata["output"],
            "model": self.metadata["model"],
            "limitations": self.metadata["limitations"],
        }
