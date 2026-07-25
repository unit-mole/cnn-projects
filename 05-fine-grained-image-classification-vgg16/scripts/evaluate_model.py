#!/usr/bin/env python
"""Evaluate a Keras model on the CIFAR-10 cat-versus-dog test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, OUTPUTS_DIR, SOURCE_MODEL_PATH
from src.dataset_loader import load_cifar10_cat_dog
from src.inference_pipeline import load_classification_model
from src.model_evaluation import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=SOURCE_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "evaluation_rerun")
    args = parser.parse_args()

    dataset = load_cifar10_cat_dog()
    model = load_classification_model(args.model)
    metrics, _, _ = evaluate_model(model, dataset, CLASS_NAMES, args.output_dir)
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
