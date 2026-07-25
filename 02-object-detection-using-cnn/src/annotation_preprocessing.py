"""Annotation helpers for this synthetic, in-memory dataset.

The source notebook generates one normalized XYXY box per image, so there are
no external VOC, COCO, or YOLO annotation files to parse.
"""

from .bounding_box_utils import sanitize_normalized_xyxy

__all__ = ["sanitize_normalized_xyxy"]
