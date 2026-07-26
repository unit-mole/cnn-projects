"""Lightweight structural validation for the GitHub Pages browser demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_STATIC_FILES = (
    "web/index.html",
    "web/assets/styles.css",
    "web/assets/app.js",
    "web/assets/model_metadata.json",
    "web/.nojekyll",
    "models/densenet121_medical_browser.h5",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-model", action="store_true")
    args = parser.parse_args()

    missing = [path for path in REQUIRED_STATIC_FILES if not Path(path).exists()]
    if missing:
        raise SystemExit(f"Missing browser-demo files: {missing}")

    html = Path("web/index.html").read_text(encoding="utf-8")
    javascript = Path("web/assets/app.js").read_text(encoding="utf-8")
    metadata = json.loads(Path("web/assets/model_metadata.json").read_text(encoding="utf-8"))

    checks = {
        "TensorFlow.js CDN": "@tensorflow/tfjs" in html,
        "model loader": "tf.loadLayersModel" in javascript,
        "medical disclaimer": "not a diagnostic tool" in html.lower(),
        "synthetic proxy disclosure": "synthetic-proxy" in html.lower(),
        "expected classes": metadata.get("classes") == ["normal_like", "pneumonia_like"],
        "expected input": metadata.get("input_shape") == [96, 96, 3],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Browser-demo validation failed: {failed}")

    if args.require_model:
        model_json = Path("web/model/model.json")
        if not model_json.exists():
            raise SystemExit("web/model/model.json was not generated")
        payload = json.loads(model_json.read_text(encoding="utf-8"))
        for group in payload.get("weightsManifest", []):
            for shard in group.get("paths", []):
                if not (model_json.parent / shard).exists():
                    raise SystemExit(f"Missing generated model shard: {shard}")

    print("GitHub Pages browser-demo validation passed")


if __name__ == "__main__":
    main()
