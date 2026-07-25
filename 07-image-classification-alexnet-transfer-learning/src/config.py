from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WEB_DIR = PROJECT_DIR / "web"


@dataclass(frozen=True)
class TrainingConfig:
    dataset: Literal["cifar10", "folder"] = "cifar10"
    model_name: Literal["alexnet", "mobilenetv2"] = "alexnet"
    image_size: int = 227
    channels: int = 3
    batch_size: int = 64
    epochs: int = 20
    validation_fraction: float = 0.15
    learning_rate: float = 1e-3
    seed: int = 42
    augment: bool = True
    use_class_weights: bool = False
    pretrained: bool = True
    fine_tune_layers: int = 0

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.image_size, self.image_size, self.channels)

    def to_dict(self) -> dict:
        return asdict(self)
