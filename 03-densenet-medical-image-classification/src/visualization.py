"""Optional plot helpers for notebook and script workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def save_training_curves(history, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for metric in ("accuracy", "loss", "roc_auc"):
        if metric not in history.history:
            continue
        figure = plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label=f"train_{metric}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            plt.plot(history.history[val_key], label=val_key)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        figure.savefig(output / f"training_{metric}.png", dpi=160)
        plt.close(figure)


def save_binary_evaluation_plots(y_true, y_score, output_dir: str | Path, threshold: float = 0.5) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    figures = [
        (ConfusionMatrixDisplay.from_predictions(y_true, y_pred).figure_, "confusion_matrix.png"),
        (RocCurveDisplay.from_predictions(y_true, y_score).figure_, "roc_curve.png"),
        (PrecisionRecallDisplay.from_predictions(y_true, y_score).figure_, "precision_recall_curve.png"),
    ]
    for figure, filename in figures:
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
