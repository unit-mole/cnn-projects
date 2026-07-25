from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.dataset_loader import load_cifar100
from src.model_evaluation import evaluate_probabilities, save_evaluation
from src.visualization import plot_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved ResNet50 model on CIFAR-100.")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "models" / "resnet50_cifar100.keras")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "outputs")
    args = parser.parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model, compile=False, safe_mode=False)
    data = load_cifar100()
    probabilities = model.predict(data.x_test, batch_size=128, verbose=1)
    metrics, report, matrix = evaluate_probabilities(data.y_test, probabilities)
    save_evaluation(args.output, metrics, report, matrix)
    plot_confusion_matrix(matrix, args.output / "figures" / "confusion_matrix.png")

    top_indices = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
    np.save(args.output / "predictions" / "top3_indices.npy", top_indices)
    print(metrics)


if __name__ == "__main__":
    main()
