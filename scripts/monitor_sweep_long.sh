#!/usr/bin/env bash
# Monitor runs/sweep_long and run aggregation when all per-run summaries are present.
# Usage: bash scripts/monitor_sweep_long.sh
TARGET=80
CHECK_INTERVAL=60   # seconds
RUN_DIR="runs/sweep_long"
OUT_DIR="runs/sweep_long/summary"

echo "Monitoring $RUN_DIR for $TARGET completed runs (experiment_summary.csv)..."
while true; do
  n=$(find "$RUN_DIR" -name "experiment_summary.csv" | wc -l)
  echo "$(date '+%F %T') - $n/$TARGET completed"
  if [ "$n" -ge "$TARGET" ]; then
    echo "Target reached. Running full aggregation..."
    python scripts/aggregate_experiment_results.py --run_dir "$RUN_DIR" --out_dir "$OUT_DIR"
    echo "Aggregation finished: $OUT_DIR"
    exit 0
  fi
  sleep "$CHECK_INTERVAL"
done
