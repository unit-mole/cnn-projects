#!/usr/bin/env python
"""Train the VGG16 classifier using a reproducible stratified split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SOURCE_MODEL_PATH, TrainingConfig
from src.dataset_loader import load_cifar10_cat_dog
from src.model_training import train_classifier
from src.vgg16_model import build_vgg16_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=SOURCE_MODEL_PATH)
    parser.add_argument("--legacy-tail-split", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
    dataset = load_cifar10_cat_dog(stratified=not args.legacy_tail_split)
    model = build_vgg16_classifier(learning_rate=config.learning_rate)
    history = train_classifier(model, dataset, output_model=args.output, config=config)
    print(f"Saved model to {args.output}")
    print(f"Completed {len(history.history.get('loss', []))} epoch(s).")


if __name__ == "__main__":
    main()
