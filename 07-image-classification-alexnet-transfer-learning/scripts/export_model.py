from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Reload and resave a Keras model for clean export.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("models/exported_model.keras"))
    args = parser.parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_path)
    print(args.output_path)


if __name__ == "__main__":
    main()
