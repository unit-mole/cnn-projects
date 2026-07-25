"""Validate the static Vercel site and TensorFlow.js model export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETERS = 471_553
EXPECTED_INPUT = [1, 64, 64, 3]
EXPECTED_OUTPUT = [1, 64, 64, 1]


def validate() -> None:
    required = [
        ROOT / "index.html",
        ROOT / "vercel.json",
        ROOT / "assets" / "css" / "styles.css",
        ROOT / "assets" / "js" / "app.js",
        ROOT / "tfjs_model" / "model.json",
        ROOT / "tfjs_model" / "weights_manifest.json",
        ROOT / "tfjs_model" / "weights.bin",
        ROOT / "tfjs_model" / "model_metadata.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Missing Vercel/TF.js files: {missing}")

    model_json = json.loads((ROOT / "tfjs_model" / "model.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / "tfjs_model" / "weights_manifest.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (ROOT / "tfjs_model" / "model_metadata.json").read_text(encoding="utf-8")
    )
    vercel_config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert model_json["format"] == "layers-model"
    assert model_json["modelTopology"]["model_config"]["class_name"] == "Functional"
    assert isinstance(manifest, list) and len(manifest) == 1
    assert manifest[0]["paths"] == ["weights.bin"]
    assert metadata["parameter_count"] == EXPECTED_PARAMETERS
    assert metadata["input_shape"] == EXPECTED_INPUT
    assert metadata["output_shape"] == EXPECTED_OUTPUT
    assert vercel_config["cleanUrls"] is True

    specs = manifest[0]["weights"]
    calculated_parameters = 0
    for spec in specs:
        count = 1
        for dimension in spec["shape"]:
            count *= int(dimension)
        calculated_parameters += count
        assert spec["dtype"] == "float32"
    assert calculated_parameters == EXPECTED_PARAMETERS

    weights = (ROOT / "tfjs_model" / "weights.bin").read_bytes()
    assert len(weights) == EXPECTED_PARAMETERS * 4
    assert len(weights) == metadata["weight_bytes"]
    assert hashlib.sha256(weights).hexdigest() == metadata["weights_sha256"]

    index_html = (ROOT / "index.html").read_text(encoding="utf-8")
    app_js = (ROOT / "assets" / "js" / "app.js").read_text(encoding="utf-8")
    assert "@tensorflow/tfjs@4.22.0" in index_html
    assert "tf.loadLayersModel" in app_js
    assert "tf.io.loadWeights" in app_js
    assert "buildUnetModel" in app_js

    for sample_number in range(1, 7):
        number = f"{sample_number:02d}"
        assert (ROOT / "assets" / "samples" / "images" / f"synthetic_tile_{number}.png").exists()
        assert (ROOT / "assets" / "samples" / "masks" / f"synthetic_tile_{number}_mask.png").exists()

    print("TensorFlow.js export validation passed.")
    print(f"Parameters: {calculated_parameters:,}")
    print(f"Weights: {len(weights):,} bytes")


if __name__ == "__main__":
    validate()
