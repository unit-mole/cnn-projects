"""Reusable source package for the ResNet50 CIFAR-100 project."""

from .class_mapping import CIFAR100_FINE_LABELS
from .config import ProjectConfig

__all__ = ["CIFAR100_FINE_LABELS", "ProjectConfig"]
