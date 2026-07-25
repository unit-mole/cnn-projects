"""Export the supplied compact U-Net from a Keras 3 archive for TensorFlow.js.

The exporter intentionally writes only inference weights, excluding optimizer state.
It produces:

* ``tfjs_model/model.json`` — conventional TensorFlow.js LayersModel metadata;
* ``tfjs_model/weights_manifest.json`` — standalone manifest used by the resilient
  browser fallback loader;
* ``tfjs_model/weights.bin`` — float32 model weights;
* ``tfjs_model/model_metadata.json`` — deployment and integrity metadata.

The browser app first tries ``tf.loadLayersModel``. If Keras/TensorFlow.js schema
compatibility differs between versions, it reconstructs the known U-Net architecture
and loads the exact same weights using ``tf.io.loadWeights``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "satellite_unet_segmentation_model.keras"
DEFAULT_OUTPUT = PROJECT_ROOT / "tfjs_model"
WEIGHTED_LAYERS = [
    "conv2d",
    "conv2d_1",
    "conv2d_2",
    "conv2d_3",
    "conv2d_4",
    "conv2d_5",
    "conv2d_6",
    "conv2d_7",
    "conv2d_8",
    "conv2d_9",
    "conv2d_10",
]


def export_model(model_path: Path, output_dir: Path) -> dict[str, object]:
    """Export a known compact U-Net Keras archive into browser assets."""
    if not model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(model_path) as archive:
        archive_names = set(archive.namelist())
        required = {"metadata.json", "config.json", "model.weights.h5"}
        missing = required - archive_names
        if missing:
            raise ValueError(f"Invalid Keras archive; missing: {sorted(missing)}")
        keras_metadata = json.loads(archive.read("metadata.json"))
        keras_config = json.loads(archive.read("config.json"))
        h5_bytes = archive.read("model.weights.h5")

    plain_specs: list[dict[str, object]] = []
    layers_model_specs: list[dict[str, object]] = []
    chunks: list[bytes] = []
    parameter_count = 0

    with tempfile.NamedTemporaryFile(suffix=".h5") as temporary_file:
        temporary_file.write(h5_bytes)
        temporary_file.flush()
        with h5py.File(temporary_file.name, "r") as h5_file:
            layers = h5_file["layers"]
            for layer_name in WEIGHTED_LAYERS:
                if layer_name not in layers:
                    raise ValueError(f"Expected layer missing from model: {layer_name}")
                variables = layers[layer_name]["vars"]
                for index, variable_name in [("0", "kernel"), ("1", "bias")]:
                    values = np.asarray(variables[index], dtype="<f4")
                    parameter_count += int(values.size)
                    plain_name = f"{layer_name}/{variable_name}"
                    spec = {
                        "name": plain_name,
                        "shape": list(values.shape),
                        "dtype": "float32",
                    }
                    plain_specs.append(spec)
                    layers_model_specs.append({**spec, "name": f"functional/{plain_name}"})
                    chunks.append(values.tobytes(order="C"))

    weights_bytes = b"".join(chunks)
    weights_path = output_dir / "weights.bin"
    weights_path.write_bytes(weights_bytes)

    standalone_manifest = [{"paths": ["weights.bin"], "weights": plain_specs}]
    (output_dir / "weights_manifest.json").write_text(
        json.dumps(standalone_manifest, indent=2), encoding="utf-8"
    )

    layers_model = {
        "format": "layers-model",
        "generatedBy": f"keras v{keras_metadata.get('keras_version', '3.x')}",
        "convertedBy": "cnn-projects deterministic Keras 3 to TensorFlow.js exporter v1",
        "modelTopology": {
            "keras_version": keras_metadata.get("keras_version", "3.x"),
            "backend": "tensorflow",
            "model_config": {
                "class_name": keras_config["class_name"],
                "config": keras_config["config"],
            },
        },
        "weightsManifest": [{"paths": ["weights.bin"], "weights": layers_model_specs}],
    }
    (output_dir / "model.json").write_text(
        json.dumps(layers_model, separators=(",", ":")), encoding="utf-8"
    )

    sha256 = hashlib.sha256(weights_bytes).hexdigest()
    metadata: dict[str, object] = {
        "project": "06-satellite-image-segmentation-unet",
        "runtime": "TensorFlow.js browser inference",
        "tfjs_version": "4.22.0",
        "model_type": "LayersModel / deterministic fallback loader",
        "architecture": "compact U-Net",
        "input_shape": [1, 64, 64, 3],
        "output_shape": [1, 64, 64, 1],
        "parameter_count": parameter_count,
        "weight_bytes": len(weights_bytes),
        "weights_sha256": sha256,
        "normalization": "RGB uint8 divided by 255.0",
        "threshold": 0.5,
        "classes": {"0": "background", "1": "synthetic urban structure"},
        "source_model": str(model_path.relative_to(PROJECT_ROOT)),
        "source_keras_version": keras_metadata.get("keras_version"),
        "deployment": "Vercel static site; inference runs entirely in the browser",
        "privacy": "Uploaded images remain in the browser and are not sent to a server.",
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export_model(args.model.resolve(), args.output.resolve())
    print(f"Exported {metadata['parameter_count']:,} parameters")
    print(f"Browser weight size: {metadata['weight_bytes']:,} bytes")
    print(f"SHA-256: {metadata['weights_sha256']}")


if __name__ == "__main__":
    main()
