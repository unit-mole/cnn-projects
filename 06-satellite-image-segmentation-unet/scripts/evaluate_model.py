from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json

from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR, PROJECT_ROOT
from src.inference_pipeline import InferencePipeline
from src.model_evaluation import evaluate_binary_masks
from src.synthetic_data import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved model on the deterministic synthetic test split.")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images, masks = generate_dataset(args.samples, args.seed)
    _, x_temp, _, y_temp = train_test_split(images, masks, test_size=0.30, random_state=args.seed)
    _, x_test, _, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=args.seed)

    pipeline = InferencePipeline(MODEL_DIR / "satellite_unet_segmentation_model.keras")
    model = pipeline.load_model()
    probabilities = model.predict(x_test, verbose=0)
    results = evaluate_binary_masks(y_test, probabilities, pipeline.config.threshold)
    output = PROJECT_ROOT / "outputs" / "metrics" / "evaluation_reproduced.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
