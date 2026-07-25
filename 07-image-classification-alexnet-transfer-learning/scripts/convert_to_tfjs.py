from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from src.model_conversion import convert_keras_to_tfjs, copy_metadata_for_web


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a trained Keras model to TensorFlow.js.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("models/model_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/tfjs_model"))
    parser.add_argument("--web-output-dir", type=Path, default=Path("web/tfjs_model"))
    parser.add_argument("--web-metadata", type=Path, default=Path("web/metadata.json"))
    args = parser.parse_args()

    summary = convert_keras_to_tfjs(args.model_path, args.output_dir, args.web_output_dir)
    copy_metadata_for_web(args.metadata, args.web_metadata)
    summary_path = Path("outputs/reports/tfjs_conversion_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
