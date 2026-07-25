"""Classification evaluation and artifact-export utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_score: list[float] = []
    for images, labels in dataset:
        raw = np.asarray(model.predict(images, verbose=0)).reshape(-1)
        y_score.extend(raw.tolist())
        y_true.extend(np.asarray(labels).reshape(-1).astype(int).tolist())
    return np.asarray(y_true), np.asarray(y_score)


def evaluate_binary_predictions(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "threshold": float(threshold),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return {"metrics": metrics, "classification_report": report, "predictions": y_pred}


def save_evaluation(result: dict, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result["metrics"], file, indent=2)
    pd.DataFrame(result["classification_report"]).transpose().to_csv(
        destination / "classification_report.csv"
    )
    pd.DataFrame(result["metrics"]["confusion_matrix"]).to_csv(
        destination / "confusion_matrix.csv", index=False
    )
