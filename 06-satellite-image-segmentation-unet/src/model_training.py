from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .unet_model import build_unet, compile_unet


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 4


def train_unet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    output_path: Path,
    config: TrainingConfig = TrainingConfig(),
):
    import tensorflow as tf

    model = compile_unet(build_unet(tuple(x_train.shape[1:])), config.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef_tf", mode="max", patience=config.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return model, history
