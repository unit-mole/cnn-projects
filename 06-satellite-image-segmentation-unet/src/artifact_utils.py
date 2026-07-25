from __future__ import annotations

import json
import zipfile
from pathlib import Path


def inspect_keras_archive(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid zipped .keras archive: {path}")
    with zipfile.ZipFile(path) as archive:
        required = {"metadata.json", "config.json", "model.weights.h5"}
        missing = required - set(archive.namelist())
        if missing:
            raise ValueError(f"Missing Keras archive members: {sorted(missing)}")
        metadata = json.loads(archive.read("metadata.json"))
        config = json.loads(archive.read("config.json"))
    return {
        "keras_metadata": metadata,
        "model_class": config.get("class_name"),
        "compile_config": config.get("compile_config"),
    }
