"""Central configuration and filesystem paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEB_DIR = PROJECT_ROOT / "web"

SOURCE_MODEL_PATH = MODELS_DIR / "vgg16_fine_grained_classification_model.keras"
BROWSER_KERAS_MODEL_PATH = MODELS_DIR / "vgg16_browser_inference.keras"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
WEB_METADATA_PATH = WEB_DIR / "metadata.json"
TFJS_MODEL_DIR = WEB_DIR / "tfjs_model"

CLASS_NAMES = ("cat", "dog")
SOURCE_IMAGE_SIZE = (32, 32)
MODEL_IMAGE_SIZE = (96, 96)
CHANNELS = 3
SIMILAR_CLASS_THRESHOLD = 0.15
SEED = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Training defaults reproduced from the supplied notebook."""

    validation_size: int = 2_000
    batch_size: int = 128
    epochs: int = 15
    learning_rate: float = 1e-3
    early_stopping_patience: int = 4
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    minimum_learning_rate: float = 1e-6
    seed: int = SEED
