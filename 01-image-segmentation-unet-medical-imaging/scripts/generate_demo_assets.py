"""Regenerate safe sample images and masks from the deterministic generator."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_data import save_demo_samples  # noqa: E402

if __name__ == "__main__":
    save_demo_samples(
        PROJECT_ROOT / "data" / "sample_images",
        PROJECT_ROOT / "data" / "sample_masks",
        count=8,
        seed=5042,
    )
    print("Synthetic demo assets generated.")
