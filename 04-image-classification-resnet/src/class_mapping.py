from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

CIFAR100_FINE_LABELS: tuple[str, ...] = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')


def validate_class_names(class_names: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(name).strip() for name in class_names)
    if not values:
        raise ValueError("At least one class name is required.")
    if any(not name for name in values):
        raise ValueError("Class names cannot be empty.")
    if len(set(values)) != len(values):
        raise ValueError("Class names must be unique.")
    return values


def label_for_index(index: int, class_names: Iterable[str] = CIFAR100_FINE_LABELS) -> str:
    names = validate_class_names(class_names)
    if index < 0 or index >= len(names):
        raise IndexError(f"Class index {index} is outside 0..{len(names)-1}.")
    return names[index]


def save_class_mapping(path: str | Path, class_names: Iterable[str] = CIFAR100_FINE_LABELS) -> Path:
    names = validate_class_names(class_names)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"index_to_class": {str(i): name for i, name in enumerate(names)}}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_class_mapping(path: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("index_to_class", payload)
    ordered = [mapping[str(i)] for i in range(len(mapping))]
    return validate_class_names(ordered)
