import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow is not installed")


def test_alexnet_output_shape():
    from src.alexnet_model import build_alexnet_style

    model = build_alexnet_style((227, 227, 3), 10)
    assert model.output_shape == (None, 10)


def test_mobilenet_output_shape_without_pretrained_download():
    from src.transfer_learning_model import build_mobilenetv2_transfer_model

    model = build_mobilenetv2_transfer_model((96, 96, 3), 10, pretrained=False)
    assert model.output_shape == (None, 10)
