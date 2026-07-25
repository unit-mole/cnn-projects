#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Starting the TensorFlow.js demo at http://localhost:8000"
python -m http.server 8000
