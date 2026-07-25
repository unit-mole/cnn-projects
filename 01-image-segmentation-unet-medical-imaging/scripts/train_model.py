"""Reproduce model training on the deterministic synthetic dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import TrainingConfig  # noqa: E402
from src.segmentation_pipeline import run_training_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
    metrics = run_training_pipeline(num_samples=args.samples, config=config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
