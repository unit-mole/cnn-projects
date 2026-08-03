from __future__ import annotations

import copy
import time
from pathlib import Path
import pandas as pd
import torch
from torch import nn


def _epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_count = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_correct += int((logits.argmax(1) == labels).sum().item())
        total_count += len(labels)
    return total_loss / total_count, total_correct / total_count


def train_model(model, train_loader, val_loader, criterion, device, epochs: int, learning_rate: float, output_path: Path, patience: int = 5):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state, best_val, stale = None, -1.0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _epoch(model, train_loader, criterion, device, optimizer, scaler)
        with torch.no_grad(): val_loss, val_acc = _epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc, "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val, stale = val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, output_path)
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping triggered.")
                break
    if best_state is not None: model.load_state_dict(best_state)
    return pd.DataFrame(history), time.perf_counter() - started
