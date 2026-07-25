from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    true_labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, labels in dataset:
        batch_probabilities = np.asarray(model.predict(images, verbose=0))
        probabilities.append(batch_probabilities)
        true_labels.append(np.asarray(labels).reshape(-1))
    return np.concatenate(true_labels), np.concatenate(probabilities)


def top_k_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int = 5) -> float:
    k = max(1, min(k, probabilities.shape[1]))
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(y_true, top, strict=True)]))


def compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Iterable[str],
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    class_names = list(class_names)
    y_true = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError("Probability matrix shape does not match class names.")

    y_pred = probabilities.argmax(axis=1)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "top_3_accuracy": top_k_accuracy(y_true, probabilities, 3),
        "top_5_accuracy": top_k_accuracy(y_true, probabilities, 5),
        "num_samples": int(len(y_true)),
    }

    try:
        binary_true = label_binarize(y_true, classes=np.arange(len(class_names)))
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(binary_true, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = None

    report_df = pd.DataFrame(report).transpose().reset_index(names="class_or_average")
    return metrics, report_df, matrix


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_training_curves(history: dict, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
        if key in history:
            ax.plot(history[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_title("Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def evaluate_and_save(
    model,
    dataset,
    class_names: list[str],
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    visualization_dir = output_dir / "visualizations"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    y_true, probabilities = collect_predictions(model, dataset)
    metrics, report_df, matrix = compute_classification_metrics(y_true, probabilities, class_names)
    (metrics_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_df.to_csv(metrics_dir / "classification_report.csv", index=False)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(metrics_dir / "confusion_matrix.csv")
    save_confusion_matrix(matrix, class_names, visualization_dir / "confusion_matrix.png")

    prediction_rows = []
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    for index, (truth, pred, conf) in enumerate(zip(y_true, predicted, confidence, strict=True)):
        prediction_rows.append(
            {
                "sample_index": index,
                "true_class": class_names[int(truth)],
                "predicted_class": class_names[int(pred)],
                "confidence": float(conf),
                "correct": bool(truth == pred),
            }
        )
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions" / "sample_predictions.csv", index=False)
    return metrics
