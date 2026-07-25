"""Compatibility wrapper matching the repository's standard project layout."""

from scripts.train_model import train


if __name__ == "__main__":
    model, history = train()
    print(f"Saved model with {model.count_params():,} parameters.")
    print(f"Recorded history keys: {', '.join(history)}")
