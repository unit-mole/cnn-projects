from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import SAMPLE_IMAGE_DIR, SAMPLE_MASK_DIR
from src.synthetic_data import make_satellite_sample


def main() -> None:
    SAMPLE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(1, 7):
        image, mask = make_satellite_sample(42 + index)
        Image.fromarray((image * 255).astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST).save(
            SAMPLE_IMAGE_DIR / f"synthetic_tile_{index:02d}.png"
        )
        Image.fromarray((mask[..., 0] * 255).astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST).save(
            SAMPLE_MASK_DIR / f"synthetic_tile_{index:02d}_mask.png"
        )
    print("Created safe synthetic demo samples.")


if __name__ == "__main__":
    main()
