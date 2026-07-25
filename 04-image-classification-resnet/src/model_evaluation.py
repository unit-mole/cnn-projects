from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

from .class_mapping import CIFAR100_FINE_LABELS


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...] = CIFAR100_FINE_LABELS,
) -> tuple[dict[str, float | None], pd.DataFrame, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError("Probability matrix does not match class mapping.")
    y_pred = probabilities.argmax(axis=1)

    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=5, labels=np.arange(len(class_names)))),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }
    try:
        one_hot = np.eye(len(class_names), dtype=np.float32)[y_true]
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = None

    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).T
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    return metrics, report, matrix


def save_evaluation(
    output_dir: str | Path,
    metrics: dict[str, Any],
    report: pd.DataFrame,
    matrix: np.ndarray,
) -> None:
    target = Path(output_dir)
    (target / "metrics").mkdir(parents=True, exist_ok=True)
    (target / "predictions").mkdir(parents=True, exist_ok=True)
    (target / "metrics" / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report.to_csv(target / "metrics" / "classification_report.csv")
    np.savetxt(target / "metrics" / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
