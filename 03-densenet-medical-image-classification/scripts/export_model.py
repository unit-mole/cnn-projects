"""Export a trained model and deployment metadata for Hugging Face Spaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["NORMAL", "PNEUMONIA"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / "densenet_medical_classification_model.keras"
    shutil.copy2(args.model, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "artifact_filename": target.name,
        "artifact_sha256": digest,
        "dataset_status": "real_dataset_requires_user_documentation",
        "classes": args.classes,
        "class_to_index": {name: index for index, name in enumerate(args.classes)},
        "input_shape": [args.input_size, args.input_size, 3],
        "external_preprocessing": "RGB resize; DenseNet preprocessing is stored inside the exported model if built by this project.",
        "medical_disclaimer": "Educational portfolio model only; not a diagnostic tool.",
    }
    (args.destination / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported {target}")


if __name__ == "__main__":
    main()
