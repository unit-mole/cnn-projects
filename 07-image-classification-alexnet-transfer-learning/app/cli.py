from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from src.inference_pipeline import load_classification_model, load_metadata, predict_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Python image classification inference.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/alexnet_cifar10.keras"))
    parser.add_argument("--metadata", type=Path, default=Path("models/model_metadata.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    model = load_classification_model(args.model)
    result = predict_class(model, args.image, metadata, args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
