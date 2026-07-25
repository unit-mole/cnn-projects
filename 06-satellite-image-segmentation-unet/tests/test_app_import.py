import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("gradio") is None, reason="Gradio is not installed")
def test_gradio_app_imports():
    import gradio_app
    assert gradio_app.demo is not None
