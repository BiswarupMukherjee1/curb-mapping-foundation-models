#!/usr/bin/env bash
# Create a virtual environment with the project dependencies (Python 3.11+, CUDA build of PyTorch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
echo "activate with: source $ROOT/.venv/bin/activate"
