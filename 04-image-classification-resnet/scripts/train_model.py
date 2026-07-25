from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.model_training import train


if __name__ == "__main__":
    model, history = train()
    print(f"Saved model with {model.count_params():,} parameters.")
    print(f"Recorded history keys: {', '.join(history)}")
