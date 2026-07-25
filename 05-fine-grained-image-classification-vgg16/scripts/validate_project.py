#!/usr/bin/env python
"""Lightweight portfolio validation that deliberately avoids model training."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_tfjs_artifacts import validate as validate_tfjs

REQUIRED = [
    "README.md",
    "README_VERCEL.md",
    "requirements.txt",
    "package.json",
    "vercel.json",
    "src/inference_pipeline.py",
    "scripts/convert_to_tfjs.py",
    "tests/test_inference_pipeline.py",
    "models/vgg16_fine_grained_classification_model.keras",
    "models/model_metadata.json",
    "web/index.html",
    "web/style.css",
    "web/app.js",
    "web/metadata.json",
    "web/tfjs_model/model.json",
]


def main() -> None:
    missing = [path for path in REQUIRED if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required project files: {missing}")

    metadata = json.loads((PROJECT_ROOT / "web/metadata.json").read_text(encoding="utf-8"))
    if metadata.get("classes") != ["cat", "dog"]:
        raise SystemExit("Unexpected browser class order.")

    package = json.loads((PROJECT_ROOT / "package.json").read_text(encoding="utf-8"))
    if package.get("scripts", {}).get("build") != "node scripts/validate-web.mjs":
        raise SystemExit("Vercel build validation command is not configured.")

    print(validate_tfjs(PROJECT_ROOT / "web/tfjs_model/model.json"))
    print("Project validation passed.")


if __name__ == "__main__":
    main()
