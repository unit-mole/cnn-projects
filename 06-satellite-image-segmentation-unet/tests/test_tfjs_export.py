from pathlib import Path

from scripts.validate_tfjs_export import validate


def test_tfjs_export_is_complete() -> None:
    validate()


def test_static_entrypoint_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "index.html").is_file()
    assert (root / "vercel.json").is_file()
