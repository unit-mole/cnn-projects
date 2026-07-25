from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def validate_tfjs_model(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    model_json = model_dir / "model.json"
    if not model_json.is_file():
        raise FileNotFoundError(f"Missing TensorFlow.js model: {model_json}")
    payload = json.loads(model_json.read_text(encoding="utf-8"))
    if payload.get("format") != "layers-model":
        raise ValueError("Expected a TensorFlow.js layers-model export.")
    topology = payload.get("modelTopology", {})
    model_config = topology.get("model_config", {})
    layers = model_config.get("config", {}).get("layers", [])
    input_layers = [layer for layer in layers if layer.get("class_name") == "InputLayer"]
    if input_layers:
        input_config = input_layers[0].get("config", {})
        if "batch_shape" in input_config:
            raise ValueError(
                "TensorFlow.js-incompatible InputLayer: replace Keras 3 'batch_shape' "
                "with 'batch_input_shape' or regenerate the model with the TFJS converter."
            )
        if not ({"batch_input_shape", "input_shape"} & input_config.keys()):
            raise ValueError(
                "TensorFlow.js InputLayer must define 'batch_input_shape' or 'input_shape'."
            )

    manifests = payload.get("weightsManifest", [])
    if not manifests:
        raise ValueError("TensorFlow.js model has no weights manifest.")
    missing = []
    shard_paths = []
    for manifest in manifests:
        for relative in manifest.get("paths", []):
            shard = model_dir / relative
            shard_paths.append(str(shard))
            if not shard.is_file():
                missing.append(str(shard))
    if missing:
        raise FileNotFoundError(f"Missing TensorFlow.js weight shards: {missing}")
    return {
        "model_json": str(model_json),
        "weight_shards": shard_paths,
        "format": payload["format"],
    }


def convert_keras_to_tfjs(
    model_path: str | Path,
    output_dir: str | Path,
    web_model_dir: str | Path | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Keras model not found: {model_path}")

    import tensorflow as tf
    import tensorflowjs as tfjs

    model = tf.keras.models.load_model(model_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tfjs.converters.save_keras_model(model, str(output_dir))
    summary = validate_tfjs_model(output_dir)

    if web_model_dir is not None:
        web_model_dir = Path(web_model_dir)
        if web_model_dir.exists():
            shutil.rmtree(web_model_dir)
        shutil.copytree(output_dir, web_model_dir)
        summary["web_model_dir"] = str(web_model_dir)

    return summary


def copy_metadata_for_web(metadata_path: str | Path, web_metadata_path: str | Path) -> Path:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    metadata["artifact_status"] = "trained"
    destination = Path(web_metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination
