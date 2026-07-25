from pathlib import Path

from scripts.validate_tfjs_artifacts import validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_web_tfjs_bundle_manifest_matches_shards():
    result = validate(PROJECT_ROOT / "web/tfjs_model/model.json")
    assert result["shards"] == 16
    assert result["weight_tensors"] == 36
    assert result["bytes"] > 60_000_000


def test_model_and_web_bundles_both_exist():
    assert (PROJECT_ROOT / "models/tfjs_model/model.json").is_file()
    assert (PROJECT_ROOT / "web/tfjs_model/model.json").is_file()
