"""Class-label utilities shared by training and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata was not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata: dict[str, Any]) -> None:
    required = {"classes", "class_to_index", "input_shape", "dataset_status"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Model metadata is missing required fields: {missing}")
    classes = metadata["classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError("metadata['classes'] must be a non-empty list.")
    expected = {name: index for index, name in enumerate(classes)}
    if metadata["class_to_index"] != expected:
        raise ValueError("class_to_index must match the order of metadata['classes'].")


def class_names(metadata: dict[str, Any]) -> list[str]:
    validate_metadata(metadata)
    return list(metadata["classes"])


def humanize_label(label: str) -> str:
    return label.replace("_", " ").strip().title()
