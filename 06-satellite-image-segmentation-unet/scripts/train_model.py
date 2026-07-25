from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR, PROJECT_ROOT
from src.model_training import TrainingConfig, train_unet
from src.synthetic_data import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact U-Net on the synthetic benchmark.")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, masks = generate_dataset(args.samples, args.seed)
    x_train, x_temp, y_train, y_temp = train_test_split(
        images, masks, test_size=0.30, random_state=args.seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=args.seed
    )
    output_path = MODEL_DIR / "satellite_unet_segmentation_model.keras"
    model, history = train_unet(
        x_train,
        y_train,
        x_val,
        y_val,
        output_path,
        TrainingConfig(epochs=args.epochs, batch_size=args.batch_size),
    )
    values = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    metrics_path = MODEL_DIR / "metrics_retrained.json"
    metrics_path.write_text(json.dumps({k: float(v) for k, v in values.items()}, indent=2), encoding="utf-8")

    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for key, val_key, filename, title in [
        ("dice_coef_tf", "val_dice_coef_tf", "dice_training_curve_retrained.png", "Dice curve"),
        ("loss", "val_loss", "loss_training_curve_retrained.png", "Loss curve"),
    ]:
        if key in history.history and val_key in history.history:
            plt.figure(figsize=(9, 4))
            plt.plot(history.history[key], label="train")
            plt.plot(history.history[val_key], label="validation")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_dir / filename, dpi=160)
            plt.close()
    print(f"Saved model: {output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
