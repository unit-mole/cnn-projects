#!/usr/bin/env python
"""Convert the browser Keras model and copy the bundle into the static web app."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BROWSER_KERAS_MODEL_PATH, MODELS_DIR, WEB_DIR
from src.model_conversion import (
    copy_tfjs_bundle,
    export_browser_keras_model,
    run_official_tfjs_converter,
    validate_tfjs_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-browser-keras-export", action="store_true")
    parser.add_argument("--quantization-bytes", type=int, choices=[1, 2], default=None)
    args = parser.parse_args()

    if not args.skip_browser_keras_export:
        _, difference = export_browser_keras_model()
        print(f"Prediction parity max absolute difference: {difference:.10g}")

    staging = MODELS_DIR / "tfjs_model_converted"
    run_official_tfjs_converter(
        BROWSER_KERAS_MODEL_PATH,
        staging,
        quantization_bytes=args.quantization_bytes,
    )
    copy_tfjs_bundle(staging, WEB_DIR / "tfjs_model")
    copy_tfjs_bundle(staging, MODELS_DIR / "tfjs_model")
    print(validate_tfjs_manifest(WEB_DIR / "tfjs_model" / "model.json"))
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
