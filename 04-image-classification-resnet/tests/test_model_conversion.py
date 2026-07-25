import json
from pathlib import Path

from src.model_conversion import validate_tfjs_bundle


def test_validate_tfjs_bundle(tmp_path: Path):
    (tmp_path / "weights.bin").write_bytes(b"1234")
    payload = {
        "format": "layers-model",
        "modelTopology": {"model_config": {}},
        "weightsManifest": [{"paths": ["weights.bin"], "weights": []}],
    }
    (tmp_path / "model.json").write_text(json.dumps(payload), encoding="utf-8")
    result = validate_tfjs_bundle(tmp_path)
    assert result["shard_count"] == 1
    assert result["has_topology"] is True
