"""Create a tiny TensorFlow.js-compatible browser wiring model.

The artifact is intentionally untrained. It verifies static hosting, model
loading, image preprocessing, inference, and result rendering before a trained
AlexNet-style model is exported. The JSON uses the Keras-v2-compatible field
names expected by TensorFlow.js Layers (not the Keras 3 ``batch_shape`` form).
"""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_SHAPE = [None, 227, 227, 3]
MODEL_NAME = "browser_smoke_test_global_rgb"

# Three global RGB features feed ten demo logits. These values are fixed and
# only make the browser workflow deterministic; they are not learned weights.
KERNEL = [
    0.8, -0.3, 0.1, 0.6, -0.2, 0.2, -0.4, 0.3, -0.1, 0.5,
    -0.2, 0.7, 0.4, -0.3, 0.8, -0.1, 0.6, 0.2, 0.4, -0.5,
    0.1, 0.2, 0.7, 0.3, 0.1, 0.8, 0.5, -0.2, 0.6, 0.4,
]
BIAS = [0.00, 0.02, -0.01, 0.01, -0.02, 0.00, 0.015, -0.015, 0.005, -0.005]


def _initializer(class_name: str) -> dict:
    return {"class_name": class_name, "config": {}}


def build_payload() -> tuple[dict, list[float]]:
    """Return a TensorFlow.js Layers manifest and its flattened float weights."""
    model_topology = {
        "keras_version": "2.15.0",
        "backend": "tensorflow",
        "model_config": {
            "class_name": "Sequential",
            "config": {
                "name": MODEL_NAME,
                "layers": [
                    {
                        "class_name": "InputLayer",
                        "config": {
                            "batch_input_shape": INPUT_SHAPE,
                            "dtype": "float32",
                            "sparse": False,
                            "name": "image",
                        },
                    },
                    {
                        "class_name": "GlobalAveragePooling2D",
                        "config": {
                            "name": "global_average_pooling",
                            "trainable": False,
                            "dtype": "float32",
                            "data_format": "channels_last",
                            "keepdims": False,
                        },
                    },
                    {
                        "class_name": "Dense",
                        "config": {
                            "name": "classifier",
                            "trainable": False,
                            "dtype": "float32",
                            "units": 10,
                            "activation": "softmax",
                            "use_bias": True,
                            "kernel_initializer": _initializer("Zeros"),
                            "bias_initializer": _initializer("Zeros"),
                            "kernel_regularizer": None,
                            "bias_regularizer": None,
                            "activity_regularizer": None,
                            "kernel_constraint": None,
                            "bias_constraint": None,
                        },
                    },
                ],
            },
        },
    }

    payload = {
        "format": "layers-model",
        "generatedBy": "Portfolio TensorFlow.js smoke-test generator; not trained",
        "convertedBy": "Manual TensorFlow.js 4.x-compatible artifact",
        "modelTopology": model_topology,
        "weightsManifest": [
            {
                "paths": ["group1-shard1of1.bin"],
                "weights": [
                    {"name": "classifier/kernel", "shape": [3, 10], "dtype": "float32"},
                    {"name": "classifier/bias", "shape": [10], "dtype": "float32"},
                ],
            }
        ],
    }
    return payload, KERNEL + BIAS


def write_model(directory: Path) -> None:
    payload, values = build_payload()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "group1-shard1of1.bin").write_bytes(
        struct.pack("<" + "f" * len(values), *values)
    )
    (directory / "SMOKE_TEST_ONLY.md").write_text(
        "# Smoke-test-only artifact\n\n"
        "This model is handcrafted and untrained. It only verifies the browser "
        "workflow. Replace it with the output of `scripts/convert_to_tfjs.py` "
        "before publishing model-performance claims.\n",
        encoding="utf-8",
    )


def main() -> None:
    models_dir = PROJECT_ROOT / "models" / "tfjs_model"
    web_dir = PROJECT_ROOT / "web" / "tfjs_model"
    for directory in (models_dir, web_dir):
        if directory.exists():
            shutil.rmtree(directory)
    write_model(models_dir)
    shutil.copytree(models_dir, web_dir)
    print(f"Created TensorFlow.js-compatible smoke-test artifacts in {models_dir} and {web_dir}")


if __name__ == "__main__":
    main()
