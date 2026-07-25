from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .alexnet_model import build_alexnet_style
from .class_mapping import save_class_mapping
from .config import MODELS_DIR, OUTPUTS_DIR, TrainingConfig
from .data_preprocessing import calculate_class_weights
from .dataset_loader import DatasetBundle, load_cifar10, load_folder_dataset
from .transfer_learning_model import build_mobilenetv2_transfer_model


def load_dataset(config: TrainingConfig, data_dir: str | Path | None = None) -> DatasetBundle:
    if config.dataset == "cifar10":
        return load_cifar10(
            image_size=config.image_size,
            batch_size=config.batch_size,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
            augment=config.augment,
        )
    if not data_dir:
        raise ValueError("--data-dir is required for a folder dataset")
    return load_folder_dataset(
        data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        augment=config.augment,
    )


def build_model(config: TrainingConfig, num_classes: int):
    if config.model_name == "alexnet":
        return build_alexnet_style(config.input_shape, num_classes, config.learning_rate)
    return build_mobilenetv2_transfer_model(
        config.input_shape,
        num_classes,
        config.learning_rate,
        pretrained=config.pretrained,
        fine_tune_layers=config.fine_tune_layers,
    )


def train_model(
    config: TrainingConfig,
    data_dir: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
    outputs_dir: str | Path = OUTPUTS_DIR,
) -> dict[str, Any]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.seed)
    models_dir = Path(models_dir)
    outputs_dir = Path(outputs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics").mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(config, data_dir)
    model = build_model(config, len(bundle.class_names))
    model_path = models_dir / f"{config.model_name}_{config.dataset}.keras"
    history_path = outputs_dir / "metrics" / f"{config.model_name}_training_history.json"
    csv_log_path = outputs_dir / "metrics" / f"{config.model_name}_training_log.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(csv_log_path),
    ]

    class_weight = calculate_class_weights(bundle.train_labels) if config.use_class_weights else None
    history = model.fit(
        bundle.train,
        validation_data=bundle.validation,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    model.save(model_path)

    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    history_path.write_text(json.dumps(history_payload, indent=2) + "\n", encoding="utf-8")
    save_class_mapping(bundle.class_names, models_dir / "class_indices.json")

    metadata = {
        "project": "07-image-classification-alexnet-transfer-learning",
        "task": "multi_class_image_classification",
        "dataset": config.dataset,
        "class_names": bundle.class_names,
        "num_classes": len(bundle.class_names),
        "input_height": config.image_size,
        "input_width": config.image_size,
        "channels": config.channels,
        "color_mode": "RGB",
        "normalization": "zero_one",
        "model_name": model.name,
        "model_family": "AlexNet-style CNN" if config.model_name == "alexnet" else "MobileNetV2 transfer-learning baseline",
        "pretrained": bool(config.pretrained and config.model_name == "mobilenetv2"),
        "artifact_status": "trained",
        "training_config": config.to_dict(),
        "responsible_use": "Educational portfolio demonstration only; not for safety-critical or production decisions.",
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metadata_path": str(models_dir / "model_metadata.json"),
        "class_mapping_path": str(models_dir / "class_indices.json"),
        "history_path": str(history_path),
        "class_names": bundle.class_names,
    }
