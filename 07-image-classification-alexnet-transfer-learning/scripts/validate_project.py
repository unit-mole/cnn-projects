from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "requirements-training.txt",
    "configs/experiment_config.json",
    "notebooks/07_pytorch_cnn_model_comparison.ipynb",
    "src/experiment_runner.py",
    "scripts/run_full_experiment.py",
    "scripts/export_to_onnx.py",
    "scripts/sync_web_evaluation.py",
    "web/index.html",
    "web/style.css",
    "web/app.js",
    "web/metadata.json",
    "web/evaluation_metrics.json",
]


def load_json(relative_path: str):
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_evaluation_assets(missing: list[str]) -> None:
    evaluation_path = ROOT / "web/evaluation_metrics.json"
    if not evaluation_path.exists():
        return
    evaluation = load_json("web/evaluation_metrics.json")
    if not evaluation.get("leaderboard"):
        missing.append("web/evaluation_metrics.json:leaderboard")
    if not evaluation.get("per_class_metrics"):
        missing.append("web/evaluation_metrics.json:per_class_metrics")
    selected_key = evaluation.get("selected_model", {}).get("key")
    if not selected_key:
        missing.append("web/evaluation_metrics.json:selected_model.key")
    for relative_asset in evaluation.get("visuals", {}).values():
        if not (ROOT / "web" / relative_asset).exists():
            missing.append(f"web/{relative_asset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-onnx", action="store_true")
    args = parser.parse_args()

    missing = [relative_path for relative_path in REQUIRED if not (ROOT / relative_path).exists()]
    load_json("configs/experiment_config.json")
    load_json("web/metadata.json")
    validate_evaluation_assets(missing)

    if args.require_onnx and not (ROOT / "web/model/model.onnx").exists():
        missing.append("web/model/model.onnx")
    if missing:
        raise SystemExit("Missing required files or data: " + ", ".join(missing))
    print(f"Project validation passed ({len(REQUIRED)} required assets plus evaluation visuals).")
