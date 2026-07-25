from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from src.model_conversion import validate_tfjs_model


REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "web/index.html",
    "web/style.css",
    "web/app.js",
    "web/metadata.json",
    "web/tfjs_model/model.json",
    "src/alexnet_model.py",
    "src/transfer_learning_model.py",
    "scripts/convert_to_tfjs.py",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    metadata = json.loads((root / "web/metadata.json").read_text(encoding="utf-8"))
    if len(metadata.get("class_names", [])) < 2:
        raise SystemExit("web/metadata.json must contain at least two class names")
    if metadata.get("normalization") not in {"zero_one", "minus_one_one", "none"}:
        raise SystemExit("Unsupported normalization in web/metadata.json")

    summary = validate_tfjs_model(root / "web/tfjs_model")
    print("Project validation passed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
