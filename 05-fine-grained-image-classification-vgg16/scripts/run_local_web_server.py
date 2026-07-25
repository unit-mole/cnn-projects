#!/usr/bin/env python
"""Serve the static TensorFlow.js app locally with correct MIME types."""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()

    web_dir = Path(__file__).resolve().parents[1] / "web"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web_dir))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.bind, args.port), handler) as server:
        print(f"Serving {web_dir} at http://{args.bind}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
