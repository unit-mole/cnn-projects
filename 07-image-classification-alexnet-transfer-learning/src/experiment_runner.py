from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .artifacts import save_json
from .class_mapping import GROUP_CLASS_NAMES
from .config import ExperimentConfig, PATHS, ensure_directories
from .dataset_loader import class_weights, load_data
from .evaluation import (
    evaluate_predictions,
    latency_profile,
    parameter_profile,
    predict_loader,
)
from .explainability import save_gradcam
from .models import build_model
from .reproducibility import runtime_environment, set_seed
from .robustness import evaluate_robustness
from .training import train_model
from .visualization import (
    save_confusion_matrix,
    save_leaderboard,
    save_prediction_gallery,
    save_training_curves,
)


def run_experiment(config_path: Path | None = None):
    config = (
        ExperimentConfig.load(config_path)
        if config_path
        else ExperimentConfig.load()
    )
    ensure_directories()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = runtime_environment()
    print("Runtime:", environment)
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable; training will run on CPU.")
    save_json(PATHS["outputs"] / "runtime_environment.json", environment)

    bundle = load_data(
        PATHS["data"],
        config.validation_size,
        config.batch_size,
        config.num_workers,
        config.image_size,
        config.seed,
    )
    save_json(PATHS["outputs"] / "dataset_summary.json", bundle.summary())

    weights = class_weights(bundle.train_labels, device)
    save_json(PATHS["outputs"] / "class_weights.json", weights.cpu().numpy())
    criterion = nn.CrossEntropyLoss(weight=weights)

    leaderboard_rows = []
    trained_models = {}

    for name in ("simple_cnn", "alexnet_style", "mobilenetv2_frozen"):
        print(f"\n===== Training {name} =====")
        model = build_model(name).to(device)
        if name == "mobilenetv2_frozen":
            model.freeze_backbone()

        specification = config.models[name]
        model_path = PATHS["models"] / f"{name}.pt"
        history, training_seconds = train_model(
            model,
            bundle.train_loader,
            bundle.val_loader,
            criterion,
            device,
            int(specification["epochs"]),
            float(specification["learning_rate"]),
            model_path,
            config.early_stopping_patience,
        )
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        trained_models[name] = model
        leaderboard_rows.append(
            _evaluate_one(
                name,
                model,
                model_path,
                history,
                training_seconds,
                bundle,
                device,
                config,
            )
        )

    print("\n===== Fine-tuning MobileNetV2 =====")
    fine_tuned = build_model("mobilenetv2_finetuned").to(device)
    fine_tuned.load_state_dict(
        torch.load(
            PATHS["models"] / "mobilenetv2_frozen.pt",
            map_location=device,
            weights_only=True,
        )
    )
    fine_tuned.unfreeze_last_blocks(
        int(config.models["mobilenetv2_finetuned"].get("unfreeze_last_blocks", 4))
    )
    specification = config.models["mobilenetv2_finetuned"]
    fine_tuned_path = PATHS["models"] / "mobilenetv2_finetuned.pt"
    history, training_seconds = train_model(
        fine_tuned,
        bundle.train_loader,
        bundle.val_loader,
        criterion,
        device,
        int(specification["epochs"]),
        float(specification["learning_rate"]),
        fine_tuned_path,
        config.early_stopping_patience,
    )
    fine_tuned.load_state_dict(
        torch.load(fine_tuned_path, map_location=device, weights_only=True)
    )
    trained_models["mobilenetv2_finetuned"] = fine_tuned
    leaderboard_rows.append(
        _evaluate_one(
            "mobilenetv2_finetuned",
            fine_tuned,
            fine_tuned_path,
            history,
            training_seconds,
            bundle,
            device,
            config,
        )
    )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        "macro_f1",
        ascending=False,
    )
    leaderboard.to_csv(PATHS["outputs"] / "model_leaderboard.csv", index=False)
    save_leaderboard(leaderboard, PATHS["outputs"] / "model_leaderboard.png")

    max_size = float(config.deployment.get("max_model_size_mb", 100))
    eligible = leaderboard[leaderboard.state_dict_size_mb <= max_size]
    selected = (eligible if len(eligible) else leaderboard).iloc[0]
    selected_name = str(selected.model)

    selected_checkpoint = PATHS["models"] / f"{selected_name}.pt"
    deployment_checkpoint = PATHS["models"] / "deployment_model.pt"
    deployment_checkpoint.write_bytes(selected_checkpoint.read_bytes())

    selection = {
        "selected_model": selected_name,
        "selection_metric": "macro_f1",
        "macro_f1": float(selected.macro_f1),
        "state_dict_size_mb": float(selected.state_dict_size_mb),
        "checkpoint": str(deployment_checkpoint.relative_to(PATHS["root"])),
    }
    save_json(PATHS["outputs"] / "selected_deployment_model.json", selection)

    selected_model = trained_models[selected_name]
    selected_output = PATHS["outputs"] / selected_name
    save_gradcam(
        selected_model,
        bundle.test_loader,
        device,
        GROUP_CLASS_NAMES,
        selected_output / "gradcam.png",
    )
    save_json(
        selected_output / "robustness_metrics.json",
        evaluate_robustness(
            selected_model,
            bundle.test_loader,
            device,
            config.robustness_sample_size,
        ),
    )

    save_json(
        PATHS["outputs"] / "experiment_summary.json",
        {
            "runtime": environment,
            "selection": selection,
            "leaderboard": leaderboard.to_dict(orient="records"),
        },
    )

    print("\nExperiment complete.")
    print("Selected deployment model:", selected_name)
    return leaderboard


def _evaluate_one(
    name,
    model,
    model_path,
    history,
    training_seconds,
    bundle,
    device,
    config,
):
    output = PATHS["outputs"] / name
    output.mkdir(parents=True, exist_ok=True)

    history.to_csv(output / "training_history.csv", index=False)
    save_training_curves(history, output / "training_accuracy.png")

    y_true, probabilities = predict_loader(model, bundle.test_loader, device)
    metrics, report, matrix, predictions = evaluate_predictions(
        y_true,
        probabilities,
        GROUP_CLASS_NAMES,
    )
    metrics.update(parameter_profile(model, model_path))
    metrics.update(
        latency_profile(
            model,
            device,
            config.image_size,
            config.latency_warmup_runs,
            config.latency_timed_runs,
        )
    )
    metrics["training_seconds"] = training_seconds

    save_json(output / "metrics.json", metrics)
    save_json(output / "classification_report.json", report)
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
    pd.DataFrame(
        matrix,
        index=GROUP_CLASS_NAMES,
        columns=GROUP_CLASS_NAMES,
    ).to_csv(output / "confusion_matrix.csv")
    save_confusion_matrix(
        matrix,
        GROUP_CLASS_NAMES,
        output / "confusion_matrix.png",
    )
    save_prediction_gallery(
        bundle.test_loader,
        y_true,
        probabilities,
        GROUP_CLASS_NAMES,
        output / "correct_predictions.png",
    )
    save_prediction_gallery(
        bundle.test_loader,
        y_true,
        probabilities,
        GROUP_CLASS_NAMES,
        output / "high_confidence_errors.png",
        wrong=True,
    )
    np.save(output / "probabilities.npy", probabilities)
    np.save(output / "predictions.npy", predictions)

    return {"model": name, **metrics}
