from __future__ import annotations

import json
import shutil

import torch

from .artifacts import save_json
from .config import ExperimentConfig, PATHS
from .dataset_loader import IMAGENET_MEAN, IMAGENET_STD
from .models import build_model


def export_selected_to_onnx(opset: int = 18):
    selection_path = PATHS["outputs"] / "selected_deployment_model.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_name = selection["selected_model"]
    config = ExperimentConfig.load()

    model = build_model(selected_name, pretrained=False)
    state_dict = torch.load(
        PATHS["models"] / "deployment_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, config.image_size, config.image_size)
    target = PATHS["models"] / "onnx_model" / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        target,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )

    web_target = PATHS["web"] / "model" / "model.onnx"
    web_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, web_target)

    metadata = {
        "selected_model": selected_name,
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [1, 3, config.image_size, config.image_size],
        "class_names": ["living", "nature", "transport", "urban"],
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "onnx_size_mb": target.stat().st_size / (1024 * 1024),
    }
    save_json(PATHS["models"] / "onnx_model" / "conversion_summary.json", metadata)
    save_json(PATHS["web"] / "model" / "conversion_summary.json", metadata)

    web_metadata_path = PATHS["web"] / "metadata.json"
    web_metadata = json.loads(web_metadata_path.read_text(encoding="utf-8"))
    web_metadata["deployment"] = {
        "selected_model": selected_name,
        "format": "ONNX",
        "status": "Ready",
        "model_path": "model/model.onnx",
    }
    web_metadata_path.write_text(
        json.dumps(web_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Exported:", target)
    print("Copied to:", web_target)
    return target
