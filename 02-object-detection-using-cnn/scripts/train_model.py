"""Training entry point.

The full, reproducible training workflow is preserved in
`notebooks/object_detection_using_cnn.ipynb`. This script intentionally does
not train automatically because the repository already includes the fitted
`.keras` artifact used by the demo.
"""

from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "object_detection_using_cnn.ipynb"

if __name__ == "__main__":
    print("Run the training notebook to reproduce the model:")
    print(NOTEBOOK)
