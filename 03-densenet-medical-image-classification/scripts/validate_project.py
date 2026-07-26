"""Lightweight project validation that avoids loading or retraining TensorFlow models."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "gradio_app.py",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "requirements-pages.txt",
    PROJECT_ROOT / "README_GITHUB_PAGES.md",
    PROJECT_ROOT / "models" / "densenet121_medical.keras",
    PROJECT_ROOT / "models" / "densenet121_medical_browser.h5",
    PROJECT_ROOT / "models" / "model_metadata.json",
    PROJECT_ROOT / "src" / "inference_pipeline.py",
    PROJECT_ROOT / "web" / "index.html",
    PROJECT_ROOT / "web" / "assets" / "app.js",
    PROJECT_ROOT / "web" / "assets" / "styles.css",
]

missing = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED if not path.exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")

with (PROJECT_ROOT / "models" / "model_metadata.json").open(encoding="utf-8") as file:
    metadata = json.load(file)
if metadata["classes"] != ["normal_like", "pneumonia_like"]:
    raise SystemExit("Unexpected class mapping in model metadata.")

keras_archive = PROJECT_ROOT / "models" / "densenet121_medical.keras"
if not zipfile.is_zipfile(keras_archive):
    raise SystemExit("The original .keras artifact is not a valid Keras v3 archive.")

browser_h5 = PROJECT_ROOT / "models" / "densenet121_medical_browser.h5"
with h5py.File(browser_h5, "r") as handle:
    if "model_config" not in handle.attrs or "model_weights" not in handle:
        raise SystemExit("The browser HDF5 artifact does not contain model topology and weights.")
    if str(handle.attrs.get("backend")) != "tensorflow":
        raise SystemExit("The browser HDF5 artifact must record tensorflow as its backend.")

print("Project structure and artifact metadata validated successfully.")
