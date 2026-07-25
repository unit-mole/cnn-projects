"""Evaluation utilities for binary and multi-class classification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_top_k_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [samples, classes].")
    k = max(1, min(int(k), probabilities.shape[1]))
    top_k = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean([truth in row for truth, row in zip(y_true, top_k)]))


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(probabilities).argmax(axis=1)
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    matrix = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "top_1_accuracy": get_top_k_accuracy(y_true, probabilities, 1),
        "top_k_accuracy": get_top_k_accuracy(y_true, probabilities, min(5, len(class_names))),
        "top_k": min(5, len(class_names)),
    }
    return metrics, report_frame, matrix


def evaluate_model(model, dataset, class_names: Sequence[str], output_dir: str | Path):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict(dataset.x_test, verbose=0)
    metrics, report, matrix = evaluate_predictions(dataset.y_test, probabilities, class_names)
    report.to_csv(output / "classification_report.csv")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output / "confusion_matrix.csv")
    pd.Series(metrics).to_json(output / "evaluation_metrics.json", indent=2)
    return metrics, report, matrix
