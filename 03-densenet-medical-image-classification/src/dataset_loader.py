"""Dataset inspection and TensorFlow folder-loader helpers."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from .config import SUPPORTED_IMAGE_SUFFIXES


def iter_image_paths(root: str | Path) -> Iterable[Path]:
    base = Path(root)
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            yield path


def inspect_folder_dataset(root: str | Path) -> dict:
    """Inspect train/val/test class folders, corrupt images, and exact duplicates."""
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory was not found: {base}")
    split_counts: dict[str, Counter] = {}
    corrupt: list[str] = []
    dimensions: Counter = Counter()
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for split in ("train", "val", "validation", "test"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        counts: Counter = Counter()
        for path in iter_image_paths(split_dir):
            class_name = path.parent.name
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions[image.size] += 1
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[digest].append(str(path))
                counts[class_name] += 1
            except (UnidentifiedImageError, OSError, ValueError):
                corrupt.append(str(path))
        split_counts[split] = counts

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "root": str(base),
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "classes": sorted({name for counts in split_counts.values() for name in counts}),
        "image_count": sum(sum(counts.values()) for counts in split_counts.values()),
        "common_dimensions": [
            {"size": list(size), "count": count} for size, count in dimensions.most_common(10)
        ],
        "corrupt_images": corrupt,
        "exact_duplicate_groups": duplicates,
    }


def make_directory_dataset(
    directory: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
):
    """Create a binary TensorFlow dataset from class subfolders."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build a training dataset.") from exc
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
