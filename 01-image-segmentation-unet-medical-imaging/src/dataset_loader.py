"""Utilities for pairing real image/mask folders if the project is extended later."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ALLOWED_IMAGE_EXTENSIONS, IMAGE_SIZE
from .image_preprocessing import preprocess_image
from .mask_preprocessing import preprocess_mask


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def pair_images_and_masks(image_dir: str | Path, mask_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match images and masks by filename stem and raise for missing pairs."""
    image_paths = _image_files(Path(image_dir))
    mask_paths = _image_files(Path(mask_dir))
    mask_by_stem = {path.stem: path for path in mask_paths}

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for image_path in image_paths:
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            missing.append(image_path.name)
        else:
            pairs.append((image_path, mask_path))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing masks for {len(missing)} image(s), including: {preview}")
    if not pairs:
        raise ValueError("No matching image-mask pairs were found.")
    return pairs


def load_pairs(
    pairs: Iterable[tuple[Path, Path]],
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for image_path, mask_path in pairs:
        image_batch, _, _ = preprocess_image(image_path, image_size=image_size)
        mask_batch = preprocess_mask(mask_path, image_size=image_size)
        images.append(image_batch[0])
        masks.append(mask_batch[0])
    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.float32)
