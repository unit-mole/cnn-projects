from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_file(path: str | Path) -> dict[str, object]:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Corrupt or unreadable image: {image_path}") from exc
    return {
        "path": str(image_path),
        "width": width,
        "height": height,
        "mode": mode,
        "sha256": image_sha256(image_path),
    }


def find_duplicate_images(paths: Iterable[str | Path]) -> dict[str, list[str]]:
    by_hash: dict[str, list[str]] = {}
    for path in paths:
        digest = image_sha256(path)
        by_hash.setdefault(digest, []).append(str(path))
    return {digest: members for digest, members in by_hash.items() if len(members) > 1}
