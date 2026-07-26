"""Validate the TensorFlow.js static deployment bundle without loading TensorFlow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_DIR / "web"
MODEL_DIR = WEB_DIR / "tfjs_model"

REQUIRED = [
    WEB_DIR / "index.html",
    WEB_DIR / "style.css",
    WEB_DIR / "app.js",
    WEB_DIR / "metadata.json",
    MODEL_DIR / "model.json",
    MODEL_DIR / "weights_manifest.json",
    MODEL_DIR / "weights.bin",
    MODEL_DIR / "model_metadata.json",
]


def main() -> None:
    missing = [str(path.relative_to(PROJECT_DIR)) for path in REQUIRED if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing deployment files: " + ", ".join(missing))

    manifest = json.loads((MODEL_DIR / "weights_manifest.json").read_text(encoding="utf-8"))
    descriptor = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
    payload = (MODEL_DIR / manifest["weight_file"]).read_bytes()

    assert len(payload) == manifest["weight_bytes"] == descriptor["weight_bytes"]
    assert hashlib.sha256(payload).hexdigest() == manifest["weight_sha256"] == descriptor["weight_sha256"]
    assert len(manifest["weights"]) == 22
    assert sum(item["value_count"] for item in manifest["weights"]) == 470_977
    assert sum(item["byte_length"] for item in manifest["weights"]) == len(payload)

    expected_offset = 0
    for item in manifest["weights"]:
        assert item["byte_offset"] == expected_offset
        assert item["byte_length"] == item["value_count"] * 4
        expected_offset += item["byte_length"]

    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "@tensorflow/tfjs@4.22.0" in html
    assert "buildCompactUnet" in javascript
    assert "weights_manifest.json" in javascript
    assert "Educational demonstration only" in html

    print(f"Vercel web validation passed: 22 tensors, {len(payload):,} weight bytes.")


if __name__ == "__main__":
    main()
