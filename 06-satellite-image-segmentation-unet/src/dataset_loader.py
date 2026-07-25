from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .image_preprocessing import SUPPORTED_SUFFIXES


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path


def _normalized_mask_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_mask", "-mask", "_label", "-label"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def pair_image_and_mask_files(image_dir: Path, mask_dir: Path) -> list[ImageMaskPair]:
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("Both image and mask directories must exist.")
    images = {
        path.stem.lower(): path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    }
    masks = {
        _normalized_mask_stem(path): path
        for path in sorted(mask_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    }
    missing = sorted(set(images) - set(masks))
    if missing:
        raise ValueError(f"Missing masks for image stems: {missing[:5]}")
    pairs = [ImageMaskPair(images[stem], masks[stem]) for stem in sorted(images)]
    if not pairs:
        raise ValueError("No matching image-mask pairs were found.")
    return pairs
