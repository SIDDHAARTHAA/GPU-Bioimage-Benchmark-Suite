#!/usr/bin/env bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
mkdir -p logs
START_TS=$(date +%s)
START_HUMAN=$(date -Is)
echo "[run] start=$START_HUMAN" > logs/four_way_bg_run.log
./scripts/run_four_way_benchmark.sh >> logs/four_way_bg_run.log 2>&1
EXIT_CODE=$?
END_TS=$(date +%s)
END_HUMAN=$(date -Is)
ELAPSED=$((END_TS-START_TS))
echo "[run] end=$END_HUMAN" >> logs/four_way_bg_run.log
echo "[run] elapsed_seconds=$ELAPSED" >> logs/four_way_bg_run.log
echo "[run] exit_code=$EXIT_CODE" >> logs/four_way_bg_run.log
