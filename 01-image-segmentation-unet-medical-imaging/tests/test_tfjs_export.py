from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "web" / "tfjs_model"


def test_browser_weight_manifest_is_complete() -> None:
    manifest = json.loads((MODEL_DIR / "weights_manifest.json").read_text(encoding="utf-8"))
    payload = (MODEL_DIR / "weights.bin").read_bytes()
    assert len(manifest["weights"]) == 22
    assert sum(item["value_count"] for item in manifest["weights"]) == 470_977
    assert sum(item["byte_length"] for item in manifest["weights"]) == len(payload)
    assert hashlib.sha256(payload).hexdigest() == manifest["weight_sha256"]


def test_vercel_static_entrypoint_exists() -> None:
    web = PROJECT_DIR / "web"
    assert (web / "index.html").exists()
    assert (web / "app.js").exists()
    assert (web / "style.css").exists()
    assert "Educational demonstration only" in (web / "index.html").read_text(encoding="utf-8")
