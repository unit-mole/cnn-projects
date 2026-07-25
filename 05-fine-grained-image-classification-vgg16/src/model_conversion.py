"""Prepare a browser-safe Keras model and convert it to TensorFlow.js.

The source model includes training-only augmentation and Keras preprocessing
operations. Browser export therefore creates a flattened inference graph whose
input is already VGG16-preprocessed 96x96 BGR data, transfers the exact learned
weights, verifies prediction parity, and then calls the official TensorFlow.js
converter when it is available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import BROWSER_KERAS_MODEL_PATH, SOURCE_MODEL_PATH, TFJS_MODEL_DIR, WEB_DIR


def build_flat_browser_model(source_model):
    import tensorflow as tf

    backbone = next(
        layer for layer in source_model.layers
        if isinstance(layer, tf.keras.Model) and "vgg16" in layer.name.lower()
    )
    head_layers = [
        layer for layer in source_model.layers
        if layer.name in {
            "flatten", "flatten_features", "dense", "dense_256",
            "batch_normalization", "dropout", "dropout_50",
            "dense_1", "dense_128", "dropout_1", "dropout_40",
            "dense_2", "predictions",
        }
    ]
    if not head_layers:
        # The supplied model uses a Sequential container after preprocessing.
        nested = [layer for layer in source_model.layers if isinstance(layer, tf.keras.Sequential)]
        candidates = [layer for container in nested for layer in container.layers]
        head_layers = [layer for layer in candidates if layer is not backbone]

    inputs = tf.keras.Input(shape=(96, 96, 3), name="vgg16_preprocessed_image")
    x = backbone(inputs, training=False)
    for layer in head_layers:
        x = layer(x, training=False) if "training" in layer.call.__code__.co_varnames else layer(x)
    browser_model = tf.keras.Model(inputs, x, name="vgg16_browser_inference")
    return browser_model


def validate_model_parity(source_model, browser_model, batches: int = 2, seed: int = 42) -> float:
    import tensorflow as tf

    generator = np.random.default_rng(seed)
    maximum_difference = 0.0
    for _ in range(batches):
        source_input = generator.random((4, 32, 32, 3), dtype=np.float32)
        source_output = np.asarray(source_model(source_input, training=False))
        resized = tf.image.resize(source_input * 255.0, (96, 96), method="bilinear")
        browser_input = tf.keras.applications.vgg16.preprocess_input(resized)
        browser_output = np.asarray(browser_model(browser_input, training=False))
        maximum_difference = max(maximum_difference, float(np.max(np.abs(source_output - browser_output))))
    return maximum_difference


def export_browser_keras_model(
    source_path: str | Path = SOURCE_MODEL_PATH,
    destination: str | Path = BROWSER_KERAS_MODEL_PATH,
) -> tuple[Path, float]:
    import tensorflow as tf

    source_model = tf.keras.models.load_model(source_path, compile=False, safe_mode=False)
    browser_model = build_flat_browser_model(source_model)
    parity_difference = validate_model_parity(source_model, browser_model)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    browser_model.save(destination)
    return destination, parity_difference


def run_official_tfjs_converter(
    keras_model: str | Path = BROWSER_KERAS_MODEL_PATH,
    output_dir: str | Path = TFJS_MODEL_DIR,
    *,
    quantization_bytes: int | None = None,
) -> Path:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "tensorflowjs.converters.converter",
        "--input_format=keras",
    ]
    if quantization_bytes in (1, 2):
        command.append(f"--quantization_bytes={quantization_bytes}")
    command.extend([str(keras_model), str(output)])
    subprocess.run(command, check=True)

    model_json = output / "model.json"
    if not model_json.exists():
        raise FileNotFoundError("TensorFlow.js converter did not create model.json.")
    return model_json


def copy_tfjs_bundle(source_dir: str | Path, destination_dir: str | Path) -> Path:
    source = Path(source_dir)
    destination = Path(destination_dir)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination / "model.json"


def validate_tfjs_manifest(model_json_path: str | Path) -> dict:
    model_json_path = Path(model_json_path)
    document = json.loads(model_json_path.read_text(encoding="utf-8"))
    if document.get("format") != "layers-model":
        raise ValueError("Expected a TensorFlow.js layers-model bundle.")
    groups = document.get("weightsManifest") or []
    if not groups:
        raise ValueError("weightsManifest is empty.")
    missing = [path for group in groups for path in group.get("paths", []) if not (model_json_path.parent / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TensorFlow.js shards: {missing}")
    return {
        "format": document["format"],
        "manifest_groups": len(groups),
        "weight_tensors": sum(len(group.get("weights", [])) for group in groups),
        "shards": sum(len(group.get("paths", [])) for group in groups),
    }
