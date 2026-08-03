from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .dataset_loader import IMAGENET_MEAN, IMAGENET_STD


def save_training_curves(history: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.epoch, history.train_accuracy, label="Train accuracy")
    ax.plot(history.epoch, history.val_accuracy, label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.epoch, history.train_loss, label="Train loss")
    ax.plot(history.epoch, history.val_loss, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_name("training_loss.png"), dpi=160)
    plt.close(fig)


def save_confusion_matrix(cm, class_names, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def denormalize(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    return np.clip(image, 0, 1)


def save_prediction_gallery(
    loader,
    y_true,
    proba,
    class_names,
    path: Path,
    wrong: bool = False,
    count: int = 12,
) -> None:
    pred = proba.argmax(1)
    confidence = proba.max(1)
    indices = np.where(pred != y_true)[0] if wrong else np.where(pred == y_true)[0]
    if wrong:
        indices = indices[np.argsort(confidence[indices])[::-1]]
    indices = indices[:count]
    if len(indices) == 0:
        return

    fig = plt.figure(figsize=(12, 8))
    for position, index in enumerate(indices, 1):
        image_tensor, _ = loader.dataset[int(index)]
        ax = fig.add_subplot(3, 4, position)
        ax.imshow(denormalize(image_tensor))
        ax.axis("off")
        ax.set_title(
            f"T:{class_names[y_true[index]]}\n"
            f"P:{class_names[pred[index]]} {confidence[index]:.1%}",
            fontsize=9,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_leaderboard(dataframe: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = dataframe.sort_values("macro_f1")
    ax.barh(ordered.model, ordered.macro_f1)
    ax.set_xlabel("Macro F1")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
