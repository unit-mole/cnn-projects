import json

import pytest

from src.class_mapping import load_class_mapping, normalize_class_mapping, save_class_mapping


def test_sequence_mapping_is_contiguous():
    assert normalize_class_mapping(["cat", "dog"]) == {0: "cat", 1: "dog"}


def test_non_contiguous_mapping_rejected():
    with pytest.raises(ValueError):
        normalize_class_mapping({0: "cat", 2: "dog"})


def test_save_and_load_mapping(tmp_path):
    target = tmp_path / "mapping.json"
    save_class_mapping(["cat", "dog"], target)
    assert load_class_mapping(target) == {0: "cat", 1: "dog"}
