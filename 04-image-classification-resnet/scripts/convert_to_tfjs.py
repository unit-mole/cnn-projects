from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.model_conversion import (
    run_official_tfjs_converter,
    synchronize_tfjs_bundle,
    validate_tfjs_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a browser-friendly Keras H5 model to TensorFlow.js.")
    parser.add_argument("--keras-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "web" / "tfjs_model")
    parser.add_argument("--quantization-bytes", type=int, choices=[1, 2])
    parser.add_argument("--sync-models-copy", action="store_true")
    args = parser.parse_args()

    run_official_tfjs_converter(
        args.keras_model,
        args.output,
        quantization_bytes=args.quantization_bytes,
    )
    summary = validate_tfjs_bundle(args.output)
    if args.sync_models_copy:
        synchronize_tfjs_bundle(args.output, PROJECT_DIR / "models" / "tfjs_model")
    summary_path = PROJECT_DIR / "outputs" / "metrics" / "tfjs_conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
