"""Create the inference-only HDF5 model used by the TensorFlow.js converter.

The original saved model accepts 28x28 RGB values scaled to [0, 1], then performs
resizing, training-only augmentation, and DenseNet preprocessing internally.
For browser deployment, those deterministic preprocessing steps are implemented in
JavaScript. This script exports a flattened 96x96 DenseNet + classification head.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

# Keras 3 can run with TensorFlow, JAX, or PyTorch. Prefer TensorFlow when present,
# while allowing the project artifact to be prepared with the PyTorch backend.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402

HEAD_LAYER_NAMES = (
    "global_average_pooling2d",
    "dense",
    "batch_normalization",
    "dropout",
    "dense_1",
)


def build_browser_model(source_model: Path) -> keras.Model:
    original = keras.models.load_model(source_model, compile=False)
    backbone = original.get_layer("densenet121")

    # Start from the backbone's own input/output so the final Functional graph is
    # flattened instead of serializing DenseNet as one opaque nested layer.
    x = backbone.output
    for name in HEAD_LAYER_NAMES:
        layer = original.get_layer(name)
        if name in {"batch_normalization", "dropout"}:
            x = layer(x, training=False)
        else:
            x = layer(x)

    return keras.Model(
        inputs=backbone.input,
        outputs=x,
        name="densenet121_medical_browser",
    )


def verify_equivalence(original_path: Path, browser_model: keras.Model) -> float:
    original = keras.models.load_model(original_path, compile=False)
    rng = np.random.default_rng(42)
    sample = rng.random((2, 28, 28, 3), dtype=np.float32)
    expected = np.asarray(original.predict(sample, verbose=0))

    resized = keras.ops.image.resize(
        keras.ops.convert_to_tensor(sample),
        (96, 96),
        interpolation="bilinear",
    )
    mean = keras.ops.convert_to_tensor([0.485, 0.456, 0.406], dtype="float32")
    std = keras.ops.convert_to_tensor([0.229, 0.224, 0.225], dtype="float32")
    preprocessed = (resized - mean) / std
    actual = np.asarray(browser_model.predict(preprocessed, verbose=0))

    maximum_difference = float(np.max(np.abs(expected - actual)))
    if maximum_difference > 1e-5:
        raise RuntimeError(
            f"Browser-model equivalence check failed: max difference={maximum_difference:.8f}"
        )
    return maximum_difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/densenet121_medical.keras"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/densenet121_medical_browser.h5"),
    )
    parser.add_argument("--skip-equivalence-check", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source model not found: {args.source}")

    browser_model = build_browser_model(args.source)
    if not args.skip_equivalence_check:
        difference = verify_equivalence(args.source, browser_model)
        print(f"Equivalence check passed; maximum absolute difference: {difference:.8g}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    browser_model.save(args.output, include_optimizer=False)

    # HDF5 is backend-neutral for this standard Layers model. TensorFlow.js expects
    # Keras/TensorFlow naming, so record tensorflow as the source backend.
    with h5py.File(args.output, "r+") as handle:
        handle.attrs["backend"] = "tensorflow"
        if "model_weights" in handle:
            handle["model_weights"].attrs["backend"] = "tensorflow"

    print(f"Browser HDF5 model written to: {args.output}")
    print(f"Parameters: {browser_model.count_params():,}")
    print(f"Input shape: {browser_model.input_shape}")
    print(f"Output shape: {browser_model.output_shape}")


if __name__ == "__main__":
    main()
