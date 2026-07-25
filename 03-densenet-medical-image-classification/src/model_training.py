"""Training helpers for frozen-backbone and fine-tuning stages."""

from __future__ import annotations

from pathlib import Path


def build_callbacks(output_dir: str | Path):
    import tensorflow as tf

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            destination / "best_model.keras",
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc", mode="max", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
        ),
        tf.keras.callbacks.CSVLogger(destination / "training_history.csv"),
    ]


def train_frozen(model, train_ds, val_ds, epochs: int, callbacks, class_weight=None):
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )
