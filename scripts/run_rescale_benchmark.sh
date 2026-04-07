#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
MPLBACKEND=Agg "$ROOT_DIR/.venv313/bin/python" "$ROOT_DIR/scripts/benchmark_cubic_rescale.py"
