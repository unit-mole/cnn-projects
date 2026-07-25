from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import functools
import http.server
import socketserver


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the static TensorFlow.js app locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--directory", type=Path, default=Path("web"))
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(f"No index.html found in {directory}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socketserver.TCPServer((args.host, args.port), handler) as server:
        print(f"Serving {directory} at http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
