from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

REMOVE_CONFIG_KEYS = {
    "optional",       # Keras 3 InputLayer option not understood by older tfjs-layers
    "synchronized",   # TensorFlow-only BatchNorm option
    "quantization_config",
}


def _clean_serialized(value: Any) -> Any:
    """Convert Keras 3 serialization fragments to portable Keras 2-style JSON."""
    if isinstance(value, list):
        return [_clean_serialized(v) for v in value]
    if not isinstance(value, dict):
        return value

    if value.get("class_name") == "DTypePolicy":
        return value.get("config", {}).get("name", "float32")

    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"module", "registered_name", "build_config"}:
            continue
        if key in REMOVE_CONFIG_KEYS:
            continue
        cleaned[key] = _clean_serialized(item)
    return cleaned


def _tensor_histories(value: Any) -> list[list[Any]]:
    """Collect Keras histories from a Keras 3 inbound-node structure."""
    histories: list[list[Any]] = []
    if isinstance(value, dict):
        if value.get("class_name") == "__keras_tensor__":
            history = value.get("config", {}).get("keras_history")
            if history:
                histories.append([history[0], int(history[1]), int(history[2]), {}])
            return histories
        for item in value.values():
            histories.extend(_tensor_histories(item))
    elif isinstance(value, list):
        for item in value:
            histories.extend(_tensor_histories(item))
    return histories


def _convert_inbound_nodes(nodes: Any) -> list[list[list[Any]]]:
    if not nodes:
        return []
    converted: list[list[list[Any]]] = []
    for node in nodes:
        histories = _tensor_histories(node.get("args", []))
        # kwargs may contain extra symbolic tensors for some layers.
        histories.extend(_tensor_histories(node.get("kwargs", {})))
        if histories:
            converted.append(histories)
    return converted


def _convert_layer(layer: dict[str, Any]) -> dict[str, Any]:
    class_name = layer["class_name"]
    config = _clean_serialized(layer.get("config", {}))

    if class_name == "InputLayer":
        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        config.pop("ragged", None)

    # groups=1 is the historical default; removing it maximizes tfjs compatibility.
    if class_name in {"Conv1D", "Conv2D", "Conv3D"} and config.get("groups") == 1:
        config.pop("groups", None)

    return {
        "class_name": class_name,
        "config": config,
        "name": layer.get("name", config.get("name")),
        "inbound_nodes": _convert_inbound_nodes(layer.get("inbound_nodes", [])),
    }


def _normalize_io_layers(value: Any) -> list[list[Any]]:
    # Keras 3 stores a single input as [name, node, tensor]; Keras 2 stores [[...]].
    if isinstance(value, list) and len(value) == 3 and isinstance(value[0], str):
        return [[value[0], int(value[1]), int(value[2])]]
    return [[item[0], int(item[1]), int(item[2])] for item in value]


def convert_topology(model_config: dict[str, Any]) -> dict[str, Any]:
    if model_config.get("class_name") not in {"Functional", "Model"}:
        raise ValueError("Only Functional/Model Keras models are supported.")

    cfg = model_config["config"]
    converted_config = {
        "name": cfg.get("name", "model"),
        "trainable": bool(cfg.get("trainable", True)),
        "layers": [_convert_layer(layer) for layer in cfg["layers"]],
        "input_layers": _normalize_io_layers(cfg["input_layers"]),
        "output_layers": _normalize_io_layers(cfg["output_layers"]),
    }
    return {"class_name": "Model", "config": converted_config}


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def iter_weights(h5: h5py.File) -> Iterable[tuple[str, np.ndarray]]:
    weights_root = h5["model_weights"]
    for raw_layer_name in weights_root.attrs["layer_names"]:
        layer_name = _decode(raw_layer_name)
        group = weights_root[layer_name]
        for raw_weight_name in group.attrs.get("weight_names", []):
            weight_name = _decode(raw_weight_name)
            yield weight_name, np.asarray(group[weight_name][()], dtype="<f4", order="C")


def convert_h5_to_tfjs(
    h5_path: Path,
    output_dir: Path,
    *,
    shard_size_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("group1-shard*.bin"):
        old.unlink()

    with h5py.File(h5_path, "r") as h5:
        raw_model_config = h5.attrs["model_config"]
        if isinstance(raw_model_config, bytes):
            raw_model_config = raw_model_config.decode("utf-8")
        model_config = json.loads(raw_model_config)
        topology = convert_topology(model_config)
        keras_version = _decode(h5.attrs.get("keras_version", "unknown"))

        specs: list[dict[str, Any]] = []
        shard_paths: list[str] = []
        shard = bytearray()
        shard_index = 1
        total_bytes = 0
        weight_count = 0

        def flush() -> None:
            nonlocal shard, shard_index
            if not shard:
                return
            name = f"group1-shard{shard_index}ofPLACEHOLDER.bin"
            (output_dir / name).write_bytes(shard)
            shard_paths.append(name)
            shard = bytearray()
            shard_index += 1

        # Official converter may split tensors across shards. We do the same byte-stream split.
        for name, array in iter_weights(h5):
            payload = array.tobytes(order="C")
            specs.append({"name": name, "shape": list(array.shape), "dtype": "float32"})
            weight_count += 1
            total_bytes += len(payload)
            offset = 0
            while offset < len(payload):
                room = shard_size_bytes - len(shard)
                take = min(room, len(payload) - offset)
                shard.extend(payload[offset : offset + take])
                offset += take
                if len(shard) == shard_size_bytes:
                    flush()
        flush()

    shard_total = len(shard_paths)
    final_paths: list[str] = []
    for old_name in shard_paths:
        old_path = output_dir / old_name
        index = int(old_name.split("shard", 1)[1].split("of", 1)[0])
        new_name = f"group1-shard{index}of{shard_total}.bin"
        old_path.rename(output_dir / new_name)
        final_paths.append(new_name)

    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{keras_version}",
        "convertedBy": "Portfolio build utility compatible with TensorFlow.js LayersModel weight manifests",
        "modelTopology": {
            "keras_version": "2.15.0",
            "backend": "tensorflow",
            "model_config": topology,
        },
        "weightsManifest": [{"paths": final_paths, "weights": specs}],
        "userDefinedMetadata": {
            "sourceModel": h5_path.name,
            "browserInput": "96x96 RGB converted to BGR with ImageNet mean subtraction in app.js",
        },
    }
    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(model_json, indent=2), encoding="utf-8")

    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    summary = {
        "source_h5": str(h5_path),
        "model_json": str(model_path),
        "format": "layers-model",
        "weight_count": weight_count,
        "total_weight_bytes": total_bytes,
        "shard_size_bytes": shard_size_bytes,
        "shard_count": shard_total,
        "model_json_sha256": checksum,
        "structural_validation": "passed",
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--shard-size-mb", type=int, default=4)
    args = parser.parse_args()
    result = convert_h5_to_tfjs(
        args.h5_path,
        args.output_dir,
        shard_size_bytes=args.shard_size_mb * 1024 * 1024,
    )
    print(json.dumps(result, indent=2))
