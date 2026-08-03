"""Publish real experiment metrics and visuals into the static web application.

Run after training and ONNX export:
    python scripts/sync_web_evaluation.py

The generated files stay inside ``web/`` so they work on GitHub Pages without a
backend or access to directories outside the published artifact.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
WEB = ROOT / "web"
EVALUATION_ASSETS = WEB / "evaluation"

MODEL_LABELS = {
    "simple_cnn": "Simple CNN",
    "alexnet_style": "AlexNet-style CNN",
    "mobilenetv2_frozen": "MobileNetV2 frozen",
    "mobilenetv2_finetuned": "MobileNetV2 fine-tuned",
}

ROBUSTNESS_LABELS = {
    "clean": "Clean images",
    "gaussian_noise": "Gaussian noise",
    "darkness": "Reduced brightness",
    "brightness": "Increased brightness",
    "blur": "Blur",
    "rotation_90": "90° rotation",
}

NUMERIC_FIELDS = {
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "top2_accuracy",
    "roc_auc_ovr_macro",
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
    "parameters_total",
    "parameters_trainable",
    "state_dict_size_mb",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "training_seconds",
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_number(field: str, value: str) -> Any:
    if field not in NUMERIC_FIELDS or value in {"", None}:
        return value
    parsed = float(value)
    if field in {"parameters_total", "parameters_trainable"}:
        return int(parsed)
    return parsed


def load_leaderboard(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Leaderboard is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {key: to_number(key, value) for key, value in row.items()}
            parsed["display_name"] = MODEL_LABELS.get(parsed["model"], parsed["model"])
            rows.append(parsed)
    rows.sort(key=lambda item: float(item["macro_f1"]), reverse=True)
    return rows


def class_metrics(report: dict[str, Any], class_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for class_name in class_names:
        values = report[class_name]
        rows.append(
            {
                "class_name": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    return rows


def robustness_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    preferred_order = ["clean", "gaussian_noise", "darkness", "brightness", "blur", "rotation_90"]
    rows = []
    for condition in preferred_order:
        if condition not in results:
            continue
        values = results[condition]
        rows.append(
            {
                "condition": condition,
                "label": ROBUSTNESS_LABELS.get(condition, condition.replace("_", " ").title()),
                "macro_f1": float(values["macro_f1"]),
                "drop_from_clean": float(values["drop_from_clean"]),
            }
        )
    return rows


def copy_visual(source: Path, target_name: str) -> str | None:
    if not source.exists():
        return None
    EVALUATION_ASSETS.mkdir(parents=True, exist_ok=True)
    destination = EVALUATION_ASSETS / target_name
    shutil.copy2(source, destination)
    return f"evaluation/{target_name}"


def update_metadata(selected_model: str) -> None:
    metadata_path = WEB / "metadata.json"
    metadata = load_json(metadata_path)
    metadata.setdefault("evaluation", {})
    metadata["evaluation"].update(
        {
            "results_file": "evaluation_metrics.json",
            "selection_metric": "macro_f1",
            "test_set_size": 10000,
        }
    )
    metadata.setdefault("deployment", {})
    metadata["deployment"].update(
        {
            "selected_model": selected_model,
            "format": "ONNX",
            "runtime": "ONNX Runtime Web",
            "status": "Model exported and validated for static GitHub Pages deployment.",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    leaderboard = load_leaderboard(OUTPUTS / "model_leaderboard.csv")
    selection = load_json(OUTPUTS / "selected_deployment_model.json")
    selected_key = str(selection["selected_model"])
    selected_row = next((row for row in leaderboard if row["model"] == selected_key), None)
    if selected_row is None:
        raise ValueError(f"Selected model {selected_key!r} is not present in the leaderboard.")

    selected_output = OUTPUTS / selected_key
    report = load_json(selected_output / "classification_report.json")
    robustness = load_json(selected_output / "robustness_metrics.json")
    metadata = load_json(WEB / "metadata.json")
    class_names = list(metadata["dataset"]["class_names"])

    visuals = {
        "leaderboard": copy_visual(OUTPUTS / "model_leaderboard.png", "model_leaderboard.png"),
        "confusion_matrix": copy_visual(selected_output / "confusion_matrix.png", "selected_confusion_matrix.png"),
        "gradcam": copy_visual(selected_output / "gradcam.png", "selected_gradcam.png"),
        "training_accuracy": copy_visual(selected_output / "training_accuracy.png", "selected_training_accuracy.png"),
        "training_loss": copy_visual(selected_output / "training_loss.png", "selected_training_loss.png"),
    }
    visuals = {key: value for key, value in visuals.items() if value is not None}

    selected_label = MODEL_LABELS.get(selected_key, selected_key)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Local RTX experiment outputs",
        "selected_model": {
            "key": selected_key,
            "display_name": selected_label,
            "selection_metric": str(selection.get("selection_metric", "macro_f1")),
            "selection_metric_value": float(selection["macro_f1"]),
            "checkpoint": str(selection.get("checkpoint", "models/deployment_model.pt")),
            "summary": (
                f"{selected_label} achieved the strongest macro F1 among the eligible candidates "
                "and was exported to ONNX for browser deployment. The AlexNet-style model remains "
                "the primary from-scratch architecture comparison."
            ),
            "metrics": selected_row,
        },
        "leaderboard": leaderboard,
        "per_class_metrics": class_metrics(report, class_names),
        "robustness": robustness_rows(robustness),
        "visuals": visuals,
        "methodology": {
            "selection_metric": "macro_f1",
            "same_split_for_all_models": True,
            "same_test_set_for_all_models": True,
            "test_set_size": 10000,
            "notes": [
                "Macro F1 is the primary ranking metric because the semantic groups are imbalanced.",
                "Latency was measured locally on the configured CUDA device and is not browser latency.",
                "The static browser demo uses the selected ONNX model through ONNX Runtime Web.",
            ],
        },
    }

    WEB.mkdir(parents=True, exist_ok=True)
    output_path = WEB / "evaluation_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    update_metadata(selected_key)

    print(f"Published evaluation data: {output_path.relative_to(ROOT)}")
    print(f"Published evaluation visuals: {EVALUATION_ASSETS.relative_to(ROOT)}")
    print(f"Selected model: {selected_label}")
    print(f"Macro F1: {float(selection['macro_f1']):.4f}")


if __name__ == "__main__":
    main()
