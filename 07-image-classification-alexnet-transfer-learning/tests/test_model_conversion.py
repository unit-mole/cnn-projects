from pathlib import Path

import json
import pytest

from src.model_conversion import validate_tfjs_model


def test_validate_tfjs_model_accepts_packaged_smoke_model():
    project = Path(__file__).resolve().parents[1]
    result = validate_tfjs_model(project / "web" / "tfjs_model")
    assert result["format"] == "layers-model"
    assert result["weight_shards"]


def test_validate_tfjs_model_rejects_missing_shard(tmp_path: Path):
    (tmp_path / "model.json").write_text(
        json.dumps({"format": "layers-model", "weightsManifest": [{"paths": ["missing.bin"], "weights": []}]}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        validate_tfjs_model(tmp_path)


def test_packaged_input_layer_is_tfjs_compatible():
    project = Path(__file__).resolve().parents[1]
    payload = json.loads((project / "web" / "tfjs_model" / "model.json").read_text(encoding="utf-8"))
    layers = payload["modelTopology"]["model_config"]["config"]["layers"]
    input_config = next(layer["config"] for layer in layers if layer["class_name"] == "InputLayer")
    assert "batch_shape" not in input_config
    assert "batch_input_shape" in input_config or "input_shape" in input_config


def test_validate_tfjs_model_rejects_keras3_batch_shape(tmp_path: Path):
    (tmp_path / "weights.bin").write_bytes(b"\x00\x00\x00\x00")
    payload = {
        "format": "layers-model",
        "modelTopology": {
            "model_config": {
                "config": {
                    "layers": [
                        {
                            "class_name": "InputLayer",
                            "config": {"batch_shape": [None, 227, 227, 3]},
                        }
                    ]
                }
            }
        },
        "weightsManifest": [{"paths": ["weights.bin"], "weights": []}],
    }
    (tmp_path / "model.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="batch_shape"):
        validate_tfjs_model(tmp_path)
