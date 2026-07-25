"""Evaluate the committed U-Net artifact on the notebook's synthetic test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, OUTPUT_DIR, SEED  # noqa: E402
from src.inference_pipeline import load_segmentation_model  # noqa: E402
from src.model_evaluation import (  # noqa: E402
    evaluate_predictions,
    intensity_threshold_baseline,
    per_sample_scores,
    predict_probabilities,
    threshold_sweep,
)
from src.metrics import dice_coefficient_np, iou_score_np  # noqa: E402
from src.synthetic_data import generate_synthetic_dataset, split_dataset  # noqa: E402


def main() -> None:
    images, masks = generate_synthetic_dataset(num_samples=2500, seed=SEED)
    _, _, x_test, _, _, y_test = split_dataset(images, masks, seed=SEED)
    model = load_segmentation_model(MODEL_PATH)
    probabilities = predict_probabilities(model, x_test)
    metrics = evaluate_predictions(y_test, probabilities)

    baseline = intensity_threshold_baseline(x_test)
    metrics["baseline_dice"] = dice_coefficient_np(y_test, baseline)
    metrics["baseline_iou"] = iou_score_np(y_test, baseline)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "re_evaluated_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    hard_masks = (probabilities >= 0.5).astype("float32")
    per_sample_scores(y_test, hard_masks).to_csv(
        OUTPUT_DIR / "re_evaluated_sample_predictions.csv", index=False
    )
    threshold_sweep(y_test, probabilities).to_csv(
        OUTPUT_DIR / "re_evaluated_threshold_sweep.csv", index=False
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
