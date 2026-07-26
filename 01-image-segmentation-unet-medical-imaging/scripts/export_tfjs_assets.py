"""Export the compact Keras 3 U-Net weights for the TensorFlow.js browser app.

This exporter intentionally avoids loading TensorFlow. It opens the supplied `.keras`
archive, extracts the inference Conv2D kernels and biases from `model.weights.h5`, and
writes a deterministic little-endian float32 bundle consumed by `web/app.js`.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
KERAS_PATH = PROJECT_DIR / "models" / "unet_medical.keras"
OUTPUT_DIR = PROJECT_DIR / "web" / "tfjs_model"
LAYER_NAMES = ["conv2d"] + [f"conv2d_{index}" for index in range(1, 11)]
EXPECTED_PARAMETER_COUNT = 470_977


def export() -> None:
    if not KERAS_PATH.exists():
        raise FileNotFoundError(f"Model not found: {KERAS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(KERAS_PATH) as archive:
        h5_payload = archive.read("model.weights.h5")

    blob = bytearray()
    manifest = []
    parameter_count = 0

    with tempfile.NamedTemporaryFile(suffix=".h5") as temporary:
        temporary.write(h5_payload)
        temporary.flush()
        with h5py.File(temporary.name, "r") as h5:
            for layer_name in LAYER_NAMES:
                group = h5[f"layers/{layer_name}/vars"]
                for index, role in ((0, "kernel"), (1, "bias")):
                    array = np.asarray(group[str(index)], dtype="<f4")
                    payload = array.tobytes(order="C")
                    offset = len(blob)
                    blob.extend(payload)
                    parameter_count += int(array.size)
                    manifest.append({
                        "name": f"{layer_name}/{role}",
                        "layer": layer_name,
                        "role": role,
                        "shape": list(array.shape),
                        "dtype": "float32",
                        "byte_offset": offset,
                        "byte_length": len(payload),
                        "value_count": int(array.size),
                    })

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"Unexpected parameter count: {parameter_count}")

    weight_path = OUTPUT_DIR / "weights.bin"
    weight_path.write_bytes(blob)
    sha256 = hashlib.sha256(blob).hexdigest()

    (OUTPUT_DIR / "weights_manifest.json").write_text(json.dumps({
        "format": "raw-little-endian-float32",
        "endianness": "little",
        "weight_file": "weights.bin",
        "weight_bytes": len(blob),
        "weight_sha256": sha256,
        "weights": manifest,
    }, indent=2) + "\n", encoding="utf-8")

    (OUTPUT_DIR / "model.json").write_text(json.dumps({
        "format": "keras-v3-custom-tfjs-bundle",
        "model_name": "medical_image_segmentation_unet",
        "input_shape": [1, 64, 64, 1],
        "output_shape": [1, 64, 64, 1],
        "parameter_count": parameter_count,
        "weight_manifest": "weights_manifest.json",
        "weight_file": "weights.bin",
        "weight_bytes": len(blob),
        "weight_sha256": sha256,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Exported {parameter_count:,} parameters")
    print(f"Weight bytes: {len(blob):,}")
    print(f"SHA-256: {sha256}")


if __name__ == "__main__":
    export()
