from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Central configuration shared by training, evaluation, and inference."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    seed: int = 42
    original_image_size: tuple[int, int] = (32, 32)
    model_image_size: tuple[int, int] = (96, 96)
    channels: int = 3
    num_classes: int = 100
    batch_size: int = 128
    epochs: int = 12
    learning_rate: float = 1e-3
    dropout_rate: float = 0.5
    dense_units: int = 512

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def keras_model_path(self) -> Path:
        return self.models_dir / "resnet50_cifar100.keras"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "model_metadata.json"
