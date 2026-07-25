"""Create the tiny, explicitly untrained TensorFlow.js browser wiring model.

This artifact exists only so the static UI can be validated before the real
AlexNet-style classifier is trained and converted. It must not be presented as
classification performance.
"""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_payload() -> tuple[dict, list[float]]:
    model_topology = {
        "keras_version": "3.0.0",
        "backend": "tensorflow",
        "model_config": {
            "class_name": "Sequential",
            "config": {
                "name": "browser_smoke_test_global_rgb",
                "trainable": False,
                "dtype": {
                    "module": "keras",
                    "class_name": "DTypePolicy",
                    "config": {"name": "float32"},
                    "registered_name": None,
                },
                "layers": [
                    {
                        "class_name": "InputLayer",
                        "config": {
                            "batch_shape": [None, 227, 227, 3],
                            "dtype": "float32",
                            "sparse": False,
                            "ragged": False,
                            "name": "image",
                        },
                    },
                    {
                        "class_name": "GlobalAveragePooling2D",
                        "config": {
                            "name": "global_average_pooling",
                            "trainable": False,
                            "dtype": {
                                "module": "keras",
                                "class_name": "DTypePolicy",
                                "config": {"name": "float32"},
                                "registered_name": None,
                            },
                            "data_format": "channels_last",
                            "keepdims": False,
                        },
                    },
                    {
                        "class_name": "Dense",
                        "config": {
                            "name": "classifier",
                            "trainable": False,
                            "dtype": {
                                "module": "keras",
                                "class_name": "DTypePolicy",
                                "config": {"name": "float32"},
                                "registered_name": None,
                            },
                            "units": 10,
                            "activation": "softmax",
                            "use_bias": True,
                            "kernel_initializer": {
                                "module": "keras.initializers",
                                "class_name": "Zeros",
                                "config": {},
                                "registered_name": None,
                            },
                            "bias_initializer": {
                                "module": "keras.initializers",
                                "class_name": "Zeros",
                                "config": {},
                                "registered_name": None,
                            },
                            "kernel_regularizer": None,
                            "bias_regularizer": None,
                            "kernel_constraint": None,
                            "bias_constraint": None,
                        },
                    },
                ],
                "build_input_shape": [None, 227, 227, 3],
            },
        },
    }
    weights = [
        0.8, -0.3, 0.1, 0.6, -0.2, 0.2, -0.4, 0.3, -0.1, 0.5,
        -0.2, 0.7, 0.4, -0.3, 0.8, -0.1, 0.6, 0.2, 0.4, -0.5,
        0.1, 0.2, 0.7, 0.3, 0.1, 0.8, 0.5, -0.2, 0.6, 0.4,
    ]
    bias = [0.00, 0.02, -0.01, 0.01, -0.02, 0.00, 0.015, -0.015, 0.005, -0.005]
    payload = {
        "format": "layers-model",
        "generatedBy": "Portfolio smoke-test generator; not trained",
        "convertedBy": "Manual TensorFlow.js-compatible artifact for interface validation",
        "modelTopology": model_topology,
        "weightsManifest": [
            {
                "paths": ["group1-shard1of1.bin"],
                "weights": [
                    {
                        "name": "browser_smoke_test_global_rgb/classifier/kernel",
                        "shape": [3, 10],
                        "dtype": "float32",
                    },
                    {
                        "name": "browser_smoke_test_global_rgb/classifier/bias",
                        "shape": [10],
                        "dtype": "float32",
                    },
                ],
            }
        ],
    }
    return payload, weights + bias


def write_model(directory: Path) -> None:
    payload, values = build_payload()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (directory / "group1-shard1of1.bin").write_bytes(struct.pack("<" + "f" * len(values), *values))
    (directory / "SMOKE_TEST_ONLY.md").write_text(
        "# Smoke-test-only artifact\n\n"
        "This model is handcrafted and untrained. Replace it with the output of "
        "`scripts/convert_to_tfjs.py` before publishing classification claims.\n",
        encoding="utf-8",
    )


def main() -> None:
    models_dir = PROJECT_ROOT / "models" / "tfjs_model"
    web_dir = PROJECT_ROOT / "web" / "tfjs_model"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    if web_dir.exists():
        shutil.rmtree(web_dir)
    write_model(models_dir)
    shutil.copytree(models_dir, web_dir)
    print(f"Created smoke-test artifacts in {models_dir} and {web_dir}")


if __name__ == "__main__":
    main()
