from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .dataset_loader import load_cifar100, one_hot_labels
from .resnet_model import build_resnet50_classifier


def default_callbacks(output_dir: Path) -> list[Any]:
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_resnet50_cifar100.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]


def train(config: ProjectConfig | None = None) -> tuple[Any, dict[str, list[float]]]:
    cfg = config or ProjectConfig()
    bundle = one_hot_labels(load_cifar100(), cfg.num_classes)
    model = build_resnet50_classifier(cfg)
    history = model.fit(
        bundle.x_train,
        bundle.y_train,
        validation_data=(bundle.x_validation, bundle.y_validation),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=default_callbacks(cfg.models_dir / "checkpoints"),
        verbose=2,
    )
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    model.save(cfg.keras_model_path)
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    (cfg.outputs_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (cfg.outputs_dir / "metrics" / "training_history.json").write_text(
        json.dumps(history_dict, indent=2), encoding="utf-8"
    )
    return model, history_dict
