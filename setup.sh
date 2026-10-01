#!/bin/bash
# One-time setup for the web app: Python packages, the trackpad lock (Swift) and the UI (TypeScript).
set -e
cd "$(dirname "$0")"

echo "1/4  Python packages (.venv)"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -q -r requirements.txt

echo "2/4  Trackpad lock (Swift)"
mkdir -p bin
swiftc -O guard/TrackpadGuard.swift -o bin/trackpad-guard

echo "3/4  Web UI (TypeScript)"
(cd ui && npm install --no-fund --no-audit --silent && npm run build --silent)

echo "4/4  Pronunciation dictionary (english/cmudict.dict, ~3.6 MB)"
[ -f english/cmudict.dict ] || .venv/bin/python pronounce.py --download
command -v espeak-ng >/dev/null || echo "     optional: brew install espeak-ng  (pronounces words the dictionary doesn't have)"

echo
echo "Done. Start the app with:  .venv/bin/python server.py"
