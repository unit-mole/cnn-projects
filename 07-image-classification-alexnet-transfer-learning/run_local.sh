#!/usr/bin/env bash
set -euo pipefail
python scripts/run_local_web_server.py --port "${PORT:-8000}"
