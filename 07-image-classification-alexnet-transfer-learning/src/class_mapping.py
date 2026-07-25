from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def validate_class_names(class_names: Iterable[str]) -> list[str]:
    cleaned = [str(name).strip() for name in class_names]
    if not cleaned:
        raise ValueError("At least one class name is required.")
    if any(not name for name in cleaned):
        raise ValueError("Class names cannot be empty.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Class names must be unique.")
    return cleaned


def build_class_mapping(class_names: Sequence[str]) -> dict[str, int]:
    names = validate_class_names(class_names)
    return {name: index for index, name in enumerate(names)}


def invert_class_mapping(mapping: dict[str, int]) -> dict[int, str]:
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Class indices must be unique.")
    return {index: name for name, index in mapping.items()}


def save_class_mapping(class_names: Sequence[str], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": validate_class_names(class_names),
        "class_to_index": build_class_mapping(class_names),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_class_mapping(source: str | Path) -> list[str]:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if "class_names" in payload:
        return validate_class_names(payload["class_names"])
    mapping = payload.get("class_to_index", payload)
    inverse = invert_class_mapping({str(k): int(v) for k, v in mapping.items()})
    return validate_class_names([inverse[i] for i in sorted(inverse)])
