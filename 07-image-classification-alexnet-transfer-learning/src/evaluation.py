from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score


def predict_loader(model, loader, device):
    model.eval(); probabilities=[]; labels=[]
    with torch.inference_mode():
        for images, y in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, 1).cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def expected_calibration_error(y_true, proba, bins=15):
    confidence = proba.max(1); prediction = proba.argmax(1); correct = prediction == y_true
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any(): ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def evaluate_predictions(y_true, proba, class_names):
    pred = proba.argmax(1)
    one_hot = np.eye(len(class_names))[y_true]
    top2 = np.argsort(proba, axis=1)[:, -2:]
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "top2_accuracy": np.mean([y_true[i] in top2[i] for i in range(len(y_true))]),
        "roc_auc_ovr_macro": roc_auc_score(one_hot, proba, average="macro", multi_class="ovr"),
        "negative_log_likelihood": log_loss(y_true, proba, labels=list(range(len(class_names)))),
        "brier_score": np.mean(np.sum((proba - one_hot) ** 2, axis=1)),
        "expected_calibration_error": expected_calibration_error(y_true, proba),
    }
    report = classification_report(y_true, pred, target_names=class_names, output_dict=True, zero_division=0)
    return metrics, report, confusion_matrix(y_true, pred), pred


def latency_profile(model, device, image_size=96, warmup=20, runs=100):
    model.eval(); sample = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        for _ in range(warmup): model(sample)
        if device.type == "cuda": torch.cuda.synchronize()
        timings=[]
        for _ in range(runs):
            start=time.perf_counter(); model(sample)
            if device.type == "cuda": torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
    return {"latency_mean_ms": float(np.mean(timings)), "latency_median_ms": float(np.median(timings)), "latency_p95_ms": float(np.percentile(timings,95))}


def parameter_profile(model, model_path: Path):
    return {
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_dict_size_mb": model_path.stat().st_size/(1024*1024) if model_path.exists() else None,
    }
