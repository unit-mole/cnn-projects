from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the TensorFlow.js demo locally.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    web_dir = Path(__file__).resolve().parents[1] / "web"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web_dir))
    url = f"http://localhost:{args.port}"
    print(f"Serving {web_dir} at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    with socketserver.TCPServer(("", args.port), handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
