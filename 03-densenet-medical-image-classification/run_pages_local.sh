#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x ".venv-pages/bin/python" ]]; then
  python3 -m venv .venv-pages
fi

source .venv-pages/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-pages.txt
python scripts/convert_browser_model.py

echo "Open http://localhost:8000 in your browser."
python -m http.server 8000 --directory web
