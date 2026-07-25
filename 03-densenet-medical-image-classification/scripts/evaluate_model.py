"""Evaluate a trained binary model against the test folder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import configure_for_performance
from src.dataset_loader import make_directory_dataset
from src.model_evaluation import collect_predictions, evaluate_binary_predictions, save_evaluation
from src.visualization import save_binary_evaluation_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/densenet_medical_classification_model.keras")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model, compile=False)
    test_ds = make_directory_dataset(args.dataset / "test", shuffle=False)
    test_ds = configure_for_performance(test_ds)
    y_true, y_score = collect_predictions(model, test_ds)
    result = evaluate_binary_predictions(y_true, y_score, args.threshold)
    save_evaluation(result, PROJECT_ROOT / "outputs" / "evaluation")
    save_binary_evaluation_plots(y_true, y_score, PROJECT_ROOT / "outputs" / "evaluation", args.threshold)
    print(result["metrics"])


if __name__ == "__main__":
    main()
