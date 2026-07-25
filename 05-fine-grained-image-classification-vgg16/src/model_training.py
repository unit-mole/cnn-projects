"""Training orchestration."""

from __future__ import annotations

from pathlib import Path

from .config import SOURCE_MODEL_PATH, TrainingConfig
from .data_preprocessing import calculate_class_weights, one_hot_encode


def build_callbacks(output_dir: str | Path, config: TrainingConfig):
    import tensorflow as tf

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output / "best_model.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.minimum_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output / "training_log.csv"),
    ]


def train_classifier(model, dataset, *, output_model: str | Path = SOURCE_MODEL_PATH, config=None):
    config = config or TrainingConfig()
    y_train = one_hot_encode(dataset.y_train, 2)
    y_validation = one_hot_encode(dataset.y_validation, 2)
    class_weights = calculate_class_weights(dataset.y_train)
    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        dataset.x_train,
        y_train,
        validation_data=(dataset.x_validation, y_validation),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=build_callbacks(output_model.parent / "checkpoints", config),
        class_weight=class_weights,
        verbose=1,
    )
    model.save(output_model)
    return history
