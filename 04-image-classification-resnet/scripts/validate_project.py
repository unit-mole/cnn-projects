from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"Validation failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project_dir.resolve()
    sys.path.insert(0, str(project))

    required = [
        "README.md",
        "README_GITHUB_PAGES.md",
        "requirements.txt",
        "web/index.html",
        "web/style.css",
        "web/app.js",
        "web/metadata.json",
        "web/tfjs_model/model.json",
        "models/model_metadata.json",
        "models/class_mapping.json",
    ]
    missing = [name for name in required if not (project / name).exists()]
    if missing:
        fail(f"missing files: {missing}")

    from src.class_mapping import CIFAR100_FINE_LABELS, load_class_mapping
    from src.model_conversion import validate_tfjs_bundle

    if len(CIFAR100_FINE_LABELS) != 100:
        fail("class label constant must contain 100 labels")
    if load_class_mapping(project / "models" / "class_mapping.json") != CIFAR100_FINE_LABELS:
        fail("class_mapping.json does not match source mapping")

    metadata = json.loads((project / "web" / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("num_classes") != 100:
        fail("web metadata num_classes must be 100")
    if metadata.get("browser_input", {}).get("height") != 96:
        fail("browser model input height must be 96")
    if len(metadata.get("class_names", [])) != 100:
        fail("web metadata must include 100 class names")

    bundle = validate_tfjs_bundle(project / "web" / "tfjs_model")
    print(json.dumps({"status": "passed", "project": str(project), "tfjs": bundle}, indent=2))


if __name__ == "__main__":
    main()
