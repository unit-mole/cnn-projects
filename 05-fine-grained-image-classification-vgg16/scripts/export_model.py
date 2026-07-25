#!/usr/bin/env python
"""Export the source model into a browser-safe Keras inference model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_conversion import export_browser_keras_model


if __name__ == "__main__":
    path, difference = export_browser_keras_model()
    print(f"Browser Keras model: {path}")
    print(f"Source/browser maximum absolute prediction difference: {difference:.10g}")
