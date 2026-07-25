from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from .object_detection_model import build_detector, compile_detector


def train_detector(
    x_train,
    y_train_class,
    y_train_box,
    x_val,
    y_val_class,
    y_val_box,
    output_path: str | Path,
    epochs: int = 12,
    batch_size: int = 64,
):
    model = compile_detector(build_detector())
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_class_output_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    history = model.fit(
        x_train,
        {"class_output": y_train_class, "box_output": y_train_box},
        validation_data=(x_val, {"class_output": y_val_class, "box_output": y_val_box}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    model.save(output_path)
    return model, history
