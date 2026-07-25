from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "cnn_detector.keras"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 1
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
