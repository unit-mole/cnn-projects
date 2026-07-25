"""End-to-end training and evaluation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from .config import MODEL_PATH, OUTPUT_DIR, SEED
from .model_evaluation import evaluate_predictions, predict_probabilities
from .model_training import TrainingConfig, save_training_artifacts, train_model
from .synthetic_data import generate_synthetic_dataset, split_dataset


def run_training_pipeline(
    num_samples: int = 2500,
    config: TrainingConfig = TrainingConfig(),
    model_path: Path = MODEL_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, float]:
    images, masks = generate_synthetic_dataset(num_samples=num_samples, seed=SEED)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(images, masks, seed=SEED)
    model, history = train_model(x_train, y_train, x_val, y_val, config=config)
    save_training_artifacts(model, history, model_path=model_path, output_dir=output_dir, config=config)
    probabilities = predict_probabilities(model, x_test)
    metrics = evaluate_predictions(y_test, probabilities)
    (output_dir / "retrained_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
