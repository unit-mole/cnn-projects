"""Build a legacy tf.keras inference model and convert it to TensorFlow.js.

Why rebuild first?
------------------
The audited source model was saved with Keras 3. TensorFlow.js converts legacy
Keras HDF5 models most reliably. This script rebuilds the standard DenseNet121
architecture with ``tf_keras``, copies every weight from the inference-only HDF5
artifact by layer name, validates a deterministic reference prediction, saves a
legacy HDF5 file, and then calls the official TensorFlow.js converter.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

# TensorFlow 2.16 defaults to Keras 3. The separately installed tf_keras package
# provides the Keras 2 serialization expected by the TensorFlow.js converter.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

EXPECTED_REFERENCE = np.array([0.7387364, 0.2612636], dtype=np.float32)


def build_legacy_model():
    try:
        import tensorflow as tf
        import tf_keras as keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow/tf_keras is not installed. Run: pip install -r requirements-pages.txt"
        ) from exc

    backbone = keras.applications.DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(96, 96, 3),
    )
    backbone.trainable = False

    x = backbone.output
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = keras.layers.Dense(2, activation="softmax", name="dense_1")(x)

    model = keras.Model(
        backbone.input,
        outputs,
        name="densenet121_medical_browser",
    )
    return tf, keras, model


def read_layer_weights(source: h5py.File, layer_name: str) -> list[np.ndarray]:
    model_weights = source["model_weights"]
    if layer_name not in model_weights:
        return []

    layer_group = model_weights[layer_name]
    weight_names = layer_group.attrs.get("weight_names", [])
    arrays: list[np.ndarray] = []
    for raw_name in weight_names:
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        arrays.append(np.asarray(layer_group[name]))
    return arrays


def copy_weights(source_h5: Path, target_model) -> tuple[int, int]:
    weighted_layers = 0
    copied_arrays = 0
    with h5py.File(source_h5, "r") as source:
        for layer in target_model.layers:
            expected = layer.get_weights()
            if not expected:
                continue

            arrays = read_layer_weights(source, layer.name)
            if not arrays:
                raise RuntimeError(f"No source weights found for layer: {layer.name}")
            if len(arrays) != len(expected):
                raise RuntimeError(
                    f"Weight-count mismatch for {layer.name}: source={len(arrays)}, target={len(expected)}"
                )
            for index, (source_array, target_array) in enumerate(zip(arrays, expected)):
                if source_array.shape != target_array.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {layer.name}[{index}]: "
                        f"source={source_array.shape}, target={target_array.shape}"
                    )

            layer.set_weights(arrays)
            weighted_layers += 1
            copied_arrays += len(arrays)

    return weighted_layers, copied_arrays


def validate_reference(tf, model) -> float:
    # This reproduces the deterministic check used when the browser artifact was
    # prepared: a seeded 28x28 RGB input, bilinear resize, then DenseNet channel
    # normalization. The expected output comes from the original audited model.
    rng = np.random.default_rng(123)
    image = rng.random((1, 28, 28, 3), dtype=np.float32)
    resized = tf.image.resize(image, [96, 96], method="bilinear")
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    preprocessed = (resized - mean) / std
    actual = np.asarray(model(preprocessed, training=False))[0]
    maximum_difference = float(np.max(np.abs(actual - EXPECTED_REFERENCE)))
    if maximum_difference > 2e-3:
        raise RuntimeError(
            "Legacy browser-model validation failed: "
            f"expected={EXPECTED_REFERENCE.tolist()}, actual={actual.tolist()}, "
            f"max_difference={maximum_difference:.8f}"
        )
    print(
        "Reference prediction passed:",
        {"expected": EXPECTED_REFERENCE.tolist(), "actual": actual.tolist()},
    )
    return maximum_difference


def run_converter(source_h5: Path, output: Path, quantization_bytes: int) -> None:
    converter = shutil.which("tensorflowjs_converter")
    if converter is None:
        raise RuntimeError(
            "tensorflowjs_converter is not installed. Run: pip install -r requirements-pages.txt"
        )

    if output.exists():
        for child in output.iterdir():
            if child.name != "README.md":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    output.mkdir(parents=True, exist_ok=True)

    command = [
        converter,
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        f"--quantization_bytes={quantization_bytes}",
        str(source_h5),
        str(output),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def validate_output(output: Path) -> tuple[int, int]:
    model_json = output / "model.json"
    if not model_json.exists():
        raise RuntimeError("TensorFlow.js conversion finished without creating model.json")

    payload = json.loads(model_json.read_text(encoding="utf-8"))
    manifests = payload.get("weightsManifest", [])
    shard_paths = [path for group in manifests for path in group.get("paths", [])]
    if not shard_paths:
        raise RuntimeError("model.json contains no weight shards")

    missing = [path for path in shard_paths if not (output / path).exists()]
    if missing:
        raise RuntimeError(f"Missing TensorFlow.js weight shards: {missing}")

    total_bytes = model_json.stat().st_size + sum(
        (output / path).stat().st_size for path in shard_paths
    )
    return len(shard_paths), total_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/densenet121_medical_browser.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/model"),
    )
    parser.add_argument(
        "--quantization-bytes",
        type=int,
        choices=(1, 2, 4),
        default=2,
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Browser HDF5 model not found: {args.source}")

    tf, _, model = build_legacy_model()
    weighted_layers, copied_arrays = copy_weights(args.source, model)
    print(f"Copied {copied_arrays} arrays across {weighted_layers} weighted layers")
    print(f"Legacy model parameters: {model.count_params():,}")
    validate_reference(tf, model)

    with tempfile.TemporaryDirectory(prefix="densenet-tfjs-") as temporary_directory:
        legacy_h5 = Path(temporary_directory) / "densenet121_medical_legacy.h5"
        model.save(legacy_h5, include_optimizer=False)
        run_converter(legacy_h5, args.output, args.quantization_bytes)

    shard_count, total_bytes = validate_output(args.output)
    print(
        f"TensorFlow.js model created with {shard_count} shard(s), "
        f"{total_bytes / 1_048_576:.2f} MiB total"
    )


if __name__ == "__main__":
    main()
