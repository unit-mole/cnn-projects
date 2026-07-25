"""Load the committed model and run one deterministic inference without retraining."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_pipeline import InferenceEngine  # noqa: E402
from src.synthetic_data import make_medical_sample  # noqa: E402


def main() -> None:
    image, _ = make_medical_sample(seed=5042)
    engine = InferenceEngine()
    result = engine.predict(image)
    assert result["mask"].size == (64, 64)
    assert 0.0 <= result["metrics"]["predicted_region_percent"] <= 100.0
    print("Model smoke test: OK")


if __name__ == "__main__":
    main()
