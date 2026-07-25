#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/run_local_web_server.py --port 8000
