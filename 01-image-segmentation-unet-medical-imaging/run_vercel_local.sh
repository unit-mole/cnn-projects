#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
node scripts/validate-web.mjs
printf '\nStarting local server at http://127.0.0.1:8000\n'
python -m http.server 8000 --directory web
