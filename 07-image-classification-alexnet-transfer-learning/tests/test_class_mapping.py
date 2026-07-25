from pathlib import Path

import pytest

from src.class_mapping import build_class_mapping, load_class_mapping, save_class_mapping, validate_class_names


def test_build_class_mapping_preserves_order():
    assert build_class_mapping(["cat", "dog"]) == {"cat": 0, "dog": 1}


def test_duplicate_classes_are_rejected():
    with pytest.raises(ValueError):
        validate_class_names(["cat", "cat"])


def test_mapping_round_trip(tmp_path: Path):
    path = save_class_mapping(["normal", "defect"], tmp_path / "classes.json")
    assert load_class_mapping(path) == ["normal", "defect"]
