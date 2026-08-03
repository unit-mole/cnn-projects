import json
from pathlib import Path


def test_web_assets():
    root = Path(__file__).resolve().parents[1]
    required = [
        "web/index.html",
        "web/app.js",
        "web/style.css",
        "web/metadata.json",
        "web/evaluation_metrics.json",
        "scripts/sync_web_evaluation.py",
    ]
    for relative_path in required:
        assert (root / relative_path).exists(), relative_path

    html = (root / "web/index.html").read_text(encoding="utf-8").lower()
    javascript = (root / "web/app.js").read_text(encoding="utf-8")
    assert "onnxruntime-web" in html
    assert "model evaluation dashboard" in html
    assert "evaluation_metrics.json" in javascript


def test_evaluation_payload():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "web/evaluation_metrics.json").read_text(encoding="utf-8"))
    assert payload["selected_model"]["key"]
    assert len(payload["leaderboard"]) == 4
    assert len(payload["per_class_metrics"]) == 4
    assert payload["selected_model"]["selection_metric"] == "macro_f1"
    for relative_asset in payload.get("visuals", {}).values():
        assert (root / "web" / relative_asset).exists(), relative_asset
