"""Class-index mapping utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .config import CLASS_MAPPING_PATH


def normalize_class_mapping(mapping: Mapping[object, object] | Sequence[str]) -> dict[int, str]:
    """Normalize a mapping or ordered label sequence into contiguous integer keys."""
    if isinstance(mapping, (list, tuple)):
        normalized = {index: str(label) for index, label in enumerate(mapping)}
    else:
        normalized = {int(index): str(label) for index, label in mapping.items()}

    expected = list(range(len(normalized)))
    if sorted(normalized) != expected:
        raise ValueError(f"Class indices must be contiguous from 0; got {sorted(normalized)}")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Class labels must be unique.")
    return dict(sorted(normalized.items()))


def load_class_mapping(path: str | Path = CLASS_MAPPING_PATH) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return normalize_class_mapping(json.load(handle))


def save_class_mapping(labels: Sequence[str], path: str | Path = CLASS_MAPPING_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mapping = normalize_class_mapping(labels)
    target.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return target
