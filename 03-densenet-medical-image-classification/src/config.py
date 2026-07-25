"""Project paths and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "data" / "sample_images"
MODEL_PATH = MODEL_DIR / "densenet121_medical.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

MEDICAL_DISCLAIMER = 'This project is for educational and portfolio demonstration purposes only. It is not a medical diagnostic tool. The model must not be used to diagnose, treat, prevent, or manage any medical condition. Medical image interpretation requires clinical validation, domain expertise, and review by qualified healthcare professionals. Do not upload private, sensitive, confidential, or personally identifiable medical images. Predictions are machine-learning outputs, not medical advice.'
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
