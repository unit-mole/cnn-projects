from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.json"


@dataclass
class ExperimentConfig:
    project_version: str = "3.0.0"
    seed: int = 42
    validation_size: int = 8000
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 96
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 2
    robustness_sample_size: int = 1500
    latency_warmup_runs: int = 20
    latency_timed_runs: int = 100
    deployment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ExperimentConfig":
        with path.open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


PATHS = {
    "root": PROJECT_ROOT,
    "data": PROJECT_ROOT / "data",
    "models": PROJECT_ROOT / "models",
    "outputs": PROJECT_ROOT / "outputs",
    "web": PROJECT_ROOT / "web",
}


def ensure_directories() -> None:
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    (PATHS["models"] / "checkpoints").mkdir(parents=True, exist_ok=True)
    (PATHS["models"] / "onnx_model").mkdir(parents=True, exist_ok=True)
    (PATHS["web"] / "model").mkdir(parents=True, exist_ok=True)
