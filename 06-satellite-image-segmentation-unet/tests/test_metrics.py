import numpy as np

from src.metrics import dice_coefficient, iou_score, precision_recall_f1


def test_perfect_masks_score_one():
    mask = np.array([[0, 1], [1, 0]], dtype=np.float32)
    assert dice_coefficient(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0
    scores = precision_recall_f1(mask, mask)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
