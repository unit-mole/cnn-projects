"""Matplotlib visualizations for training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_training_curves(history, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    values = history.history if hasattr(history, "history") else history

    for metric in ("accuracy", "loss"):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(values.get(metric, []), label=f"training {metric}")
        axis.plot(values.get(f"val_{metric}", []), label=f"validation {metric}")
        axis.set_title(f"Training and validation {metric}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.title())
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / f"training_{metric}_curve.png", dpi=180)
        plt.close(figure)


def save_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
