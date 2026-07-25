from __future__ import annotations

from pathlib import Path

from .config import MODELS_DIR, OUTPUTS_DIR, WEB_DIR, TrainingConfig
from .dataset_loader import load_cifar10, load_folder_dataset
from .model_conversion import convert_keras_to_tfjs, copy_metadata_for_web
from .model_evaluation import evaluate_and_save
from .model_training import train_model


def run_end_to_end(
    config: TrainingConfig,
    data_dir: str | Path | None = None,
    convert_for_web: bool = True,
) -> dict:
    training = train_model(config, data_dir)

    if config.dataset == "cifar10":
        bundle = load_cifar10(config.image_size, config.batch_size, config.validation_fraction, config.seed, False)
    else:
        bundle = load_folder_dataset(data_dir, config.image_size, config.batch_size, config.validation_fraction, config.seed, False)

    import tensorflow as tf

    model = tf.keras.models.load_model(training["model_path"])
    metrics = evaluate_and_save(model, bundle.test, bundle.class_names, OUTPUTS_DIR)

    conversion = None
    if convert_for_web:
        conversion = convert_keras_to_tfjs(
            training["model_path"],
            MODELS_DIR / "tfjs_model",
            WEB_DIR / "tfjs_model",
        )
        copy_metadata_for_web(training["metadata_path"], WEB_DIR / "metadata.json")

    return {"training": training, "metrics": metrics, "conversion": conversion}
