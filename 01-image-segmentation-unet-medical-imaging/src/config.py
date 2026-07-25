"""Central configuration for training, evaluation, and inference."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

MODEL_PATH = MODEL_DIR / "unet_medical.keras"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

IMAGE_SIZE = (64, 64)
INPUT_CHANNELS = 1
MASK_THRESHOLD = 0.50
BASELINE_THRESHOLD = 0.55
SEED = 42
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MEDICAL_DISCLAIMER = 'This project is for educational and portfolio demonstration purposes only. It is not a medical diagnostic tool. The model must not be used to diagnose, treat, prevent, or manage any medical condition. Medical image interpretation requires clinical validation, domain expertise, and review by qualified healthcare professionals. Do not upload private, sensitive, confidential, or personally identifiable medical images. Predicted masks are machine-learning outputs, not medical advice.'
