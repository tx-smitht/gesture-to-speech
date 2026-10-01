#!/bin/bash
# One-time setup for the web app: Python packages, the trackpad lock (Swift) and the UI (TypeScript).
set -e
cd "$(dirname "$0")"

echo "1/3  Python packages (.venv)"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -q -r requirements.txt

echo "2/3  Trackpad lock (Swift)"
mkdir -p bin
swiftc -O guard/TrackpadGuard.swift -o bin/trackpad-guard

echo "3/3  Web UI (TypeScript)"
(cd ui && npm install --no-fund --no-audit --silent && npm run build --silent)

echo
echo "Done. Start the app with:  .venv/bin/python server.py"
