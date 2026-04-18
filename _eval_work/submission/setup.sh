#!/bin/bash
set -e

echo "[setup] Installing dependencies..."
pip install -r "$(dirname "$0")/requirements.txt"
echo "[setup] Done."
