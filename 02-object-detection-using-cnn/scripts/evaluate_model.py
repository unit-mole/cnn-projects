import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
metrics = json.loads((PROJECT_DIR / "models" / "metrics.json").read_text(encoding="utf-8"))

if __name__ == "__main__":
    print(json.dumps(metrics, indent=2))
