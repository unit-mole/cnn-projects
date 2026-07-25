from __future__ import annotations

import numpy as np


def sanitize_normalized_xyxy(box: np.ndarray | list[float]) -> np.ndarray:
    """Clip, order, and validate a normalized [x1, y1, x2, y2] box."""
    arr = np.asarray(box, dtype=np.float32).reshape(4)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    x1, y1, x2, y2 = np.clip(arr, 0.0, 1.0)
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return np.asarray([left, top, right, bottom], dtype=np.float32)


def normalized_to_pixel_xyxy(
    box: np.ndarray | list[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = sanitize_normalized_xyxy(box)
    px1 = int(round(x1 * max(width - 1, 1)))
    py1 = int(round(y1 * max(height - 1, 1)))
    px2 = int(round(x2 * max(width - 1, 1)))
    py2 = int(round(y2 * max(height - 1, 1)))
    return px1, py1, px2, py2


def box_area_xyxy(box: np.ndarray | list[float]) -> float:
    x1, y1, x2, y2 = sanitize_normalized_xyxy(box)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def iou_xyxy(
    true_box: np.ndarray | list[float],
    predicted_box: np.ndarray | list[float],
) -> float:
    tx1, ty1, tx2, ty2 = sanitize_normalized_xyxy(true_box)
    px1, py1, px2, py2 = sanitize_normalized_xyxy(predicted_box)

    ix1, iy1 = max(tx1, px1), max(ty1, py1)
    ix2, iy2 = min(tx2, px2), min(ty2, py2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    union = box_area_xyxy(true_box) + box_area_xyxy(predicted_box) - intersection
    return float(intersection / union) if union > 0 else 0.0
