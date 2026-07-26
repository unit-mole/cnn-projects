"""Convert the inference-only DenseNet model to TensorFlow.js.

The source browser model was exported by Keras 3, which stores layer names such
as ``conv1_conv``.  The legacy ``tf_keras`` DenseNet implementation used by the
TensorFlow.js converter names the same layers ``conv1/conv``.  This module maps
those equivalent names safely, validates every weight shape, rebuilds a legacy
Keras HDF5 model, checks a deterministic reference prediction, and then runs the
official TensorFlow.js converter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

# TensorFlow 2.16 defaults to Keras 3. The separately installed tf_keras package
# provides the Keras 2 serialization expected by TensorFlow.js 4.x.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

EXPECTED_REFERENCE = np.array([0.7387364, 0.2612636], dtype=np.float32)


def build_legacy_model():
    """Build the legacy tf_keras architecture expected by the converter."""
    try:
        import tensorflow as tf
        import tf_keras as keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow/tf_keras is not installed. "
            "Run: pip install -r requirements-pages.txt"
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


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def canonical_layer_name(name: str) -> str:
    """Normalize equivalent Keras 2/Keras 3 layer-name separators.

    Examples:
        ``conv1/conv`` -> ``conv1_conv``
        ``conv1_conv`` -> ``conv1_conv``
    """
    normalized = re.sub(r"[/\\]+", "_", name.strip())
    return re.sub(r"_+", "_", normalized)


def read_layer_weights(source: h5py.File, layer_name: str) -> list[np.ndarray]:
    """Read all arrays for one Keras HDF5 layer group."""
    model_weights = source["model_weights"]
    if layer_name not in model_weights:
        return []

    layer_group = model_weights[layer_name]
    weight_names = layer_group.attrs.get("weight_names", [])
    arrays: list[np.ndarray] = []
    for raw_name in weight_names:
        name = _decode(raw_name)
        arrays.append(np.asarray(layer_group[name]))
    return arrays


def source_weight_manifest(source: h5py.File) -> list[tuple[str, list[np.ndarray]]]:
    """Return weighted source layers in their serialized model order."""
    model_weights = source["model_weights"]
    raw_layer_names: Iterable[object] = model_weights.attrs.get(
        "layer_names", list(model_weights.keys())
    )

    manifest: list[tuple[str, list[np.ndarray]]] = []
    for raw_name in raw_layer_names:
        name = _decode(raw_name)
        arrays = read_layer_weights(source, name)
        if arrays:
            manifest.append((name, arrays))
    return manifest


def _same_shapes(arrays: list[np.ndarray], expected: list[np.ndarray]) -> bool:
    return len(arrays) == len(expected) and all(
        source_array.shape == target_array.shape
        for source_array, target_array in zip(arrays, expected)
    )


def _shape_summary(arrays: list[np.ndarray]) -> list[tuple[int, ...]]:
    return [tuple(array.shape) for array in arrays]


def copy_weights(source_h5: Path, target_model) -> tuple[int, int]:
    """Copy Keras 3 HDF5 weights into the legacy tf_keras model.

    Resolution strategy:
    1. Exact layer-name match.
    2. Canonical match after converting ``/`` separators to ``_``.
    3. Same-position fallback in the weighted-layer manifest, accepted only when
       all array counts and shapes match.

    The third step protects against additional harmless naming differences while
    still refusing to copy incompatible weights.
    """
    weighted_layers = 0
    copied_arrays = 0

    with h5py.File(source_h5, "r") as source:
        manifest = source_weight_manifest(source)
        if not manifest:
            raise RuntimeError("The source HDF5 file contains no serialized weights")

        by_exact = {name: arrays for name, arrays in manifest}
        by_canonical: dict[str, list[tuple[str, list[np.ndarray]]]] = {}
        for name, arrays in manifest:
            by_canonical.setdefault(canonical_layer_name(name), []).append((name, arrays))

        target_weighted_layers = [
            layer for layer in target_model.layers if layer.get_weights()
        ]
        if len(target_weighted_layers) != len(manifest):
            raise RuntimeError(
                "Weighted-layer count mismatch: "
                f"source={len(manifest)}, target={len(target_weighted_layers)}. "
                "The source and rebuilt DenseNet architectures are not equivalent."
            )

        used_source_names: set[str] = set()

        for position, layer in enumerate(target_weighted_layers):
            expected = layer.get_weights()
            selected_name: str | None = None
            selected_arrays: list[np.ndarray] | None = None
            selection_method = ""

            # 1. Exact match.
            arrays = by_exact.get(layer.name)
            if arrays is not None and layer.name not in used_source_names:
                if _same_shapes(arrays, expected):
                    selected_name = layer.name
                    selected_arrays = arrays
                    selection_method = "exact"

            # 2. Keras 2 slash names versus Keras 3 underscore names.
            if selected_arrays is None:
                candidates = by_canonical.get(canonical_layer_name(layer.name), [])
                compatible = [
                    (name, candidate_arrays)
                    for name, candidate_arrays in candidates
                    if name not in used_source_names
                    and _same_shapes(candidate_arrays, expected)
                ]
                if len(compatible) == 1:
                    selected_name, selected_arrays = compatible[0]
                    selection_method = "canonical-name"
                elif len(compatible) > 1:
                    raise RuntimeError(
                        f"Ambiguous source layer mapping for {layer.name}: "
                        f"{[name for name, _ in compatible]}"
                    )

            # 3. Serialized weighted-layer order, guarded by exact shapes.
            if selected_arrays is None:
                fallback_name, fallback_arrays = manifest[position]
                if (
                    fallback_name not in used_source_names
                    and _same_shapes(fallback_arrays, expected)
                ):
                    selected_name = fallback_name
                    selected_arrays = fallback_arrays
                    selection_method = "ordered-shape-fallback"

            if selected_arrays is None or selected_name is None:
                fallback_name, fallback_arrays = manifest[position]
                raise RuntimeError(
                    "Unable to map source weights for target layer "
                    f"{layer.name!r} at weighted position {position}. "
                    f"Target shapes={_shape_summary(expected)}; "
                    f"same-position source={fallback_name!r} "
                    f"with shapes={_shape_summary(fallback_arrays)}"
                )

            layer.set_weights(selected_arrays)
            used_source_names.add(selected_name)
            weighted_layers += 1
            copied_arrays += len(selected_arrays)

            if selection_method != "exact":
                print(
                    f"Mapped target layer {layer.name!r} <- source layer "
                    f"{selected_name!r} ({selection_method})"
                )

        if len(used_source_names) != len(manifest):
            unused = [name for name, _ in manifest if name not in used_source_names]
            raise RuntimeError(f"Not all source weighted layers were used: {unused}")

    return weighted_layers, copied_arrays


def validate_reference(tf, model) -> float:
    """Confirm that rebuilt-model predictions match the audited artifact."""
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
    """Run the installed official tensorflowjs_converter command."""
    converter = shutil.which("tensorflowjs_converter")
    if converter is None:
        raise RuntimeError(
            "tensorflowjs_converter is not installed. "
            "Run: pip install -r requirements-pages.txt"
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
    """Validate model.json and every referenced binary weight shard."""
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
        model.save(str(legacy_h5), include_optimizer=False)
        run_converter(legacy_h5, args.output, args.quantization_bytes)

    shard_count, total_bytes = validate_output(args.output)
    print(
        f"TensorFlow.js model created with {shard_count} shard(s), "
        f"{total_bytes / 1_048_576:.2f} MiB total"
    )


if __name__ == "__main__":
    main()
