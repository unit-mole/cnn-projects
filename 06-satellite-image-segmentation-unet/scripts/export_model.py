from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import shutil
from pathlib import Path

from src.artifact_utils import inspect_keras_archive
from src.config import MODEL_METADATA_PATH, MODEL_PATH, PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal Hugging Face Space bundle.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "huggingface_space_bundle")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    inspect_keras_archive(MODEL_PATH)
    for name in ["app.py", "gradio_app.py", "requirements.txt", "README.md"]:
        shutil.copy2(PROJECT_ROOT / name, destination / name)
    shutil.copytree(PROJECT_ROOT / "src", destination / "src")
    shutil.copytree(PROJECT_ROOT / "models", destination / "models")
    shutil.copytree(PROJECT_ROOT / "data" / "sample_images", destination / "data" / "sample_images")
    summary = {"bundle": str(destination), "model_metadata": json.loads(MODEL_METADATA_PATH.read_text())}
    (destination / "BUNDLE_INFO.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Created Hugging Face bundle: {destination}")


if __name__ == "__main__":
    main()
