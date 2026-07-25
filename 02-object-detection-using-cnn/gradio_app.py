"""Compatibility entry point for users who prefer `python gradio_app.py`."""

from app import demo

if __name__ == "__main__":
    demo.launch()
