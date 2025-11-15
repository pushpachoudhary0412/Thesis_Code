#!/usr/bin/env python3
"""
Run detectors for runs listed in runs/sweep_long/detector_outputs/missing_runs.json.

This script is intended to be started with the detector venv python:
.venv_detectors/bin/python scripts/run_missing.py

It runs sequentially and writes per-run logs (detect_log_{method}.txt) in each run dir.
"""
import json
import subprocess
from pathlib import Path
import sys
import time
import os
from typing import List

ROOT = Path.cwd()
MISSING = ROOT / "runs" / "sweep_long" / "detector_outputs" / "missing_runs.json"
BATCH_SIZE = 256
TOP_K = 1
METHODS: List[str] = ["spectral", "activation_clustering"]

def load_missing():
    with MISSING.open("r") as f:
        return json.load(f)

def run_for_run(rel: str):
    run_dir = ROOT / "runs" / "sweep_long" / rel
    if not run_dir.exists():
        print(f"[SKIP] run dir not found: {run_dir}")
        return
    for method in METHODS:
        target = run_dir / f"results_detect_{method}.json"
        if target.exists():
            print(f"[SKIP] {method} already exists for {rel}")
            continue
        log_path = run_dir / f"detect_log_{method}.txt"
        cmd = [
            sys.executable,
            "-m",
            "mimiciv_backdoor_study.detect",
            "--run_dir", str(run_dir),
            "--method", method,
            "--batch_size", str(BATCH_SIZE),
        ]
        if method == "spectral":
            cmd += ["--top_k", str(TOP_K)]
        with open(log_path, "a") as lf:
            lf.write(f"=== START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            lf.write("Running: " + " ".join(cmd) + "\n\n")
            lf.flush()
            try:
                rc = subprocess.run(cmd, stdout=lf, stderr=lf, env=dict(PYTHONPATH=str(ROOT)), timeout=None)
                lf.write(f"\n=== EXIT {rc.returncode} ===\n\n")
            except Exception as e:
                lf.write(f"\nERROR: {e}\\n\\n")
    # small sleep to avoid hammering resources on loops
    time.sleep(0.5)

def main():
    miss = load_missing()
    print(f"Will process {len(miss)} runs")
    for rel in miss:
        try:
            print(f"Processing {rel}")
            run_for_run(rel)
        except KeyboardInterrupt:
            print("Interrupted by user, exiting.")
            break
        except Exception as e:
            print(f"Unhandled exception for {rel}: {e}")
    print("Done processing missing runs.")

if __name__ == '__main__':
    main()
