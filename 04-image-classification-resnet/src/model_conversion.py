from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_official_tfjs_converter(
    keras_h5_path: str | Path,
    output_dir: str | Path,
    *,
    quantization_bytes: int | None = None,
) -> Path:
    source = Path(keras_h5_path)
    target = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    converter = shutil.which("tensorflowjs_converter")
    command = (
        [converter, "--input_format=keras"]
        if converter
        else [sys.executable, "-m", "tensorflowjs.converters.converter", "--input_format=keras"]
    )
    if quantization_bytes in {1, 2}:
        command.append(f"--quantization_bytes={quantization_bytes}")
    command.extend([str(source), str(target)])
    subprocess.run(command, check=True)
    validate_tfjs_bundle(target)
    return target / "model.json"


def validate_tfjs_bundle(model_dir: str | Path) -> dict[str, Any]:
    directory = Path(model_dir)
    model_path = directory / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if payload.get("format") != "layers-model":
        raise ValueError("Expected a TensorFlow.js layers-model bundle.")
    manifests = payload.get("weightsManifest")
    if not manifests:
        raise ValueError("model.json does not contain a weightsManifest.")
    missing: list[str] = []
    empty: list[str] = []
    shard_count = 0
    for group in manifests:
        for relative in group.get("paths", []):
            shard_count += 1
            path = directory / relative
            if not path.exists():
                missing.append(relative)
            elif path.stat().st_size == 0:
                empty.append(relative)
    if missing or empty:
        raise ValueError(f"Invalid TensorFlow.js shards. Missing={missing}, empty={empty}")
    topology = payload.get("modelTopology", {})
    return {
        "format": payload["format"],
        "shard_count": shard_count,
        "has_topology": bool(topology),
        "model_json_bytes": model_path.stat().st_size,
    }


def synchronize_tfjs_bundle(source_dir: str | Path, destination_dir: str | Path) -> None:
    source = Path(source_dir)
    destination = Path(destination_dir)
    validate_tfjs_bundle(source)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
