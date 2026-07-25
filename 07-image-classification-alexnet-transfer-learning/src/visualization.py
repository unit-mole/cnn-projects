from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_class_distribution(labels: np.ndarray, class_names: list[str], destination: str | Path) -> Path:
    labels = np.asarray(labels).reshape(-1)
    counts = np.bincount(labels, minlength=len(class_names))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, counts)
    ax.set_ylabel("Images")
    ax.set_title("Class Distribution")
    ax.tick_params(axis="x", rotation=45)
    for index, value in enumerate(counts):
        ax.text(index, value, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_prediction_gallery(images, true_labels, predicted_labels, class_names, destination: str | Path, limit: int = 12) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = np.asarray(images)[:limit]
    true_labels = np.asarray(true_labels).reshape(-1)[:limit]
    predicted_labels = np.asarray(predicted_labels).reshape(-1)[:limit]
    columns = 4
    rows = int(np.ceil(len(images) / columns))
    fig = plt.figure(figsize=(12, 3 * rows))
    for index, image in enumerate(images):
        ax = fig.add_subplot(rows, columns, index + 1)
        ax.imshow(np.clip(image, 0, 1))
        ax.axis("off")
        ax.set_title(f"T: {class_names[int(true_labels[index])]}\nP: {class_names[int(predicted_labels[index])]}")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination
