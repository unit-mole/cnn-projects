from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path

from src.artifact_utils import inspect_keras_archive
from src.config import MODEL_METADATA_PATH, MODEL_PATH, PROJECT_ROOT

REQUIRED = [
    "app.py",
    "gradio_app.py",
    "README.md",
    "README_HUGGINGFACE.md",
    "README_VERCEL.md",
    "index.html",
    "vercel.json",
    "tfjs_model/model.json",
    "tfjs_model/weights.bin",
    "requirements.txt",
    "models/satellite_unet_segmentation_model.keras",
    "models/model_metadata.json",
    "notebooks/satellite_image_segmentation_unet_kaggle.ipynb",
]


def main() -> None:
    missing = [item for item in REQUIRED if not (PROJECT_ROOT / item).exists()]
    if missing:
        raise SystemExit(f"Missing required project files: {missing}")
    archive = inspect_keras_archive(MODEL_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["input"]["height"] == 64
    assert metadata["input"]["width"] == 64
    assert metadata["output"]["mask_type"] == "binary"
    assert archive["model_class"] == "Functional"
    print("Project validation passed.")


if __name__ == "__main__":
    main()
