"""Train a real folder-based chest X-ray model after obtaining the dataset legally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import configure_for_performance
from src.dataset_loader import inspect_folder_dataset, make_directory_dataset
from src.densenet_model import DenseNetConfig, build_densenet121, unfreeze_for_fine_tuning
from src.model_training import build_callbacks, train_frozen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Folder containing train/val/test class folders.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    audit = inspect_folder_dataset(args.dataset)
    print(json.dumps(audit, indent=2))
    train_ds = make_directory_dataset(args.dataset / "train", (args.image_size, args.image_size), args.batch_size)
    val_name = "val" if (args.dataset / "val").exists() else "validation"
    val_ds = make_directory_dataset(args.dataset / val_name, (args.image_size, args.image_size), args.batch_size, shuffle=False)
    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    model, backbone = build_densenet121(DenseNetConfig(image_size=(args.image_size, args.image_size)))
    callbacks = build_callbacks(PROJECT_ROOT / "outputs" / "training")
    train_frozen(model, train_ds, val_ds, args.epochs, callbacks)
    if args.fine_tune_epochs > 0:
        unfreeze_for_fine_tuning(model, backbone)
        train_frozen(model, train_ds, val_ds, args.fine_tune_epochs, callbacks)
    destination = PROJECT_ROOT / "models" / "densenet_medical_classification_model.keras"
    model.save(destination)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
