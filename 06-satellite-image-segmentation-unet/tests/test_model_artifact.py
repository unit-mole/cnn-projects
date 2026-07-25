from src.artifact_utils import inspect_keras_archive
from src.config import MODEL_PATH


def test_model_is_valid_keras_archive():
    info = inspect_keras_archive(MODEL_PATH)
    assert info["model_class"] == "Functional"
    assert "keras_version" in info["keras_metadata"]
