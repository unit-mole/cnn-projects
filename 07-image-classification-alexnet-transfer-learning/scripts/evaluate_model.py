from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from src.class_mapping import load_class_mapping
from src.dataset_loader import load_cifar10, load_folder_dataset
from src.model_evaluation import evaluate_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained image-classification model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cifar10", "folder"], default="cifar10")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--class-mapping", type=Path, default=Path("models/class_indices.json"))
    parser.add_argument("--image-size", type=int, default=227)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    import tensorflow as tf

    class_names = load_class_mapping(args.class_mapping)
    if args.dataset == "cifar10":
        bundle = load_cifar10(args.image_size, args.batch_size, augment=False)
    else:
        if not args.data_dir:
            parser.error("--data-dir is required for a folder dataset")
        bundle = load_folder_dataset(args.data_dir, args.image_size, args.batch_size, augment=False)
    if class_names != bundle.class_names:
        raise ValueError("Class mapping does not match the loaded dataset.")

    model = tf.keras.models.load_model(args.model_path)
    metrics = evaluate_and_save(model, bundle.test, class_names, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
