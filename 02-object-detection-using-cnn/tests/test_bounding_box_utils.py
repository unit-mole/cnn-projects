import numpy as np

from src.bounding_box_utils import iou_xyxy, normalized_to_pixel_xyxy, sanitize_normalized_xyxy


def test_sanitize_orders_and_clips_box():
    result = sanitize_normalized_xyxy([1.2, 0.8, -0.2, 0.1])
    np.testing.assert_allclose(result, [0.0, 0.1, 1.0, 0.8])


def test_identical_boxes_have_iou_one():
    assert iou_xyxy([0.1, 0.2, 0.7, 0.9], [0.1, 0.2, 0.7, 0.9]) == 1.0


def test_pixel_conversion_stays_in_bounds():
    box = normalized_to_pixel_xyxy([0, 0, 1, 1], width=64, height=64)
    assert box == (0, 0, 63, 63)
