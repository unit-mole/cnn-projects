from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "satellite_unet_segmentation_model.keras"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
RUNTIME_DIR = PROJECT_ROOT / "outputs" / "runtime"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "data" / "sample_images"
SAMPLE_MASK_DIR = PROJECT_ROOT / "data" / "sample_masks"


@dataclass(frozen=True)
class InferenceConfig:
    height: int = 64
    width: int = 64
    channels: int = 3
    threshold: float = 0.5
    overlay_alpha: float = 0.45


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_inference_config() -> InferenceConfig:
    metadata = load_json(MODEL_METADATA_PATH)
    return InferenceConfig(
        height=int(metadata["input"]["height"]),
        width=int(metadata["input"]["width"]),
        channels=int(metadata["input"]["channels"]),
        threshold=float(metadata["output"]["threshold"]),
    )
