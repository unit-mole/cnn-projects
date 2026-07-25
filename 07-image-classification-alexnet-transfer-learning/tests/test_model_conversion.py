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
