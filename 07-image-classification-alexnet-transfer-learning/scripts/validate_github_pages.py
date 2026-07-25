"""Validate files required by the GitHub Pages browser deployment."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
REQUIRED_FILES = (
    "index.html",
    "style.css",
    "app.js",
    "metadata.json",
    "tfjs_model/model.json",
)


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (WEB_ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing GitHub Pages files: {', '.join(missing)}")

    metadata = json.loads((WEB_ROOT / "metadata.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((WEB_ROOT / "tfjs_model/model.json").read_text(encoding="utf-8"))

    if metadata.get("primary_hosting") != "GitHub Pages":
        raise SystemExit("metadata.json must identify GitHub Pages as primary hosting.")
    if "modelTopology" not in model_manifest and "modelTopologyBytes" not in model_manifest:
        raise SystemExit("TensorFlow.js model manifest is missing model topology information.")

    shard_names = {
        path
        for group in model_manifest.get("weightsManifest", [])
        for path in group.get("paths", [])
    }
    missing_shards = [
        path for path in sorted(shard_names) if not (WEB_ROOT / "tfjs_model" / path).is_file()
    ]
    if missing_shards:
        raise SystemExit(f"Missing TensorFlow.js shards: {', '.join(missing_shards)}")

    print("GitHub Pages browser deployment validation passed.")


if __name__ == "__main__":
    main()
