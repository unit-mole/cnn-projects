"""Lightweight validation that avoids loading or retraining TensorFlow models."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
required = [
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "gradio_app.py",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "models" / "densenet121_medical.keras",
    PROJECT_ROOT / "models" / "model_metadata.json",
    PROJECT_ROOT / "src" / "inference_pipeline.py",
]
missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")

with (PROJECT_ROOT / "models" / "model_metadata.json").open(encoding="utf-8") as file:
    metadata = json.load(file)
if metadata["classes"] != ["normal_like", "pneumonia_like"]:
    raise SystemExit("Unexpected class mapping.")
if not zipfile.is_zipfile(PROJECT_ROOT / "models" / "densenet121_medical.keras"):
    raise SystemExit("The .keras artifact is not a valid Keras v3 archive.")

print("Project structure and artifact metadata validated successfully.")
