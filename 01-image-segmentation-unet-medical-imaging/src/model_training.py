"""Training helpers for the reproducible synthetic U-Net experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MODEL_PATH, OUTPUT_DIR
from .unet_model import build_unet


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 4
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6


def train_model(
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    config: TrainingConfig = TrainingConfig(),
) -> tuple[Any, Any]:
    import tensorflow as tf

    model = build_unet(
        input_shape=tuple(x_train.shape[1:]),
        learning_rate=config.learning_rate,
        compile_model=True,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef_tf",
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def save_training_artifacts(
    model: Any,
    history: Any,
    model_path: Path = MODEL_PATH,
    output_dir: Path = OUTPUT_DIR,
    config: TrainingConfig = TrainingConfig(),
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "training_config.json").write_text(
        __import__("json").dumps(asdict(config), indent=2), encoding="utf-8"
    )
