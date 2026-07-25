from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from src.config import TrainingConfig
from src.model_training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an AlexNet-style model or MobileNetV2 baseline.")
    parser.add_argument("--dataset", choices=["cifar10", "folder"], default="cifar10")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model", choices=["alexnet", "mobilenetv2"], default="alexnet")
    parser.add_argument("--image-size", type=int, default=227)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fine-tune-layers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        dataset=args.dataset,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_fraction=args.validation_fraction,
        learning_rate=args.learning_rate,
        seed=args.seed,
        augment=not args.no_augment,
        use_class_weights=args.class_weights,
        pretrained=not args.no_pretrained,
        fine_tune_layers=args.fine_tune_layers,
    )
    result = train_model(config, args.data_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
