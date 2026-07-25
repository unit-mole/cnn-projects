"""Alternate local entrypoint retained for portfolio structure compatibility."""

from app import APP_CSS, APP_THEME, demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, theme=APP_THEME, css=APP_CSS)
