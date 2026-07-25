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


def _input_layer_config(model_manifest: dict) -> dict:
    topology = model_manifest.get("modelTopology", {})
    model_config = topology.get("model_config", {})
    config = model_config.get("config", {})
    layers = config.get("layers", [])
    for layer in layers:
        if layer.get("class_name") == "InputLayer":
            return layer.get("config", {})
    raise SystemExit("TensorFlow.js model manifest has no InputLayer.")


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

    input_config = _input_layer_config(model_manifest)
    if "batch_shape" in input_config:
        raise SystemExit(
            "InputLayer uses Keras 3 'batch_shape'; TensorFlow.js Layers requires "
            "'batch_input_shape' or 'input_shape'."
        )
    if not ({"batch_input_shape", "input_shape"} & input_config.keys()):
        raise SystemExit(
            "InputLayer must define 'batch_input_shape' or 'input_shape' for TensorFlow.js."
        )

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

    app_source = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    if "createBrowserSmokeTestModel" not in app_source:
        raise SystemExit("app.js is missing the browser smoke-model fallback.")

    print("GitHub Pages browser deployment validation passed.")


if __name__ == "__main__":
    main()
