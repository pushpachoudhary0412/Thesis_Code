#!/usr/bin/env python3
"""
Benchmark runner for detectors.

Usage (example):
  # using local python (assumes required deps installed in active env)
  PYTHONPATH=$(pwd) python scripts/benchmark.py --poison_rates 0.0 0.1 0.2 --triggers none rare_value --seeds 42 43 --detector activation_clustering --epochs 2

  # using a named conda env (recommended on macOS for pyarrow)
  PYTHONPATH=$(pwd) python scripts/benchmark.py --conda_env mimiciv_env --poison_rates 0.0 0.1 --triggers none --seeds 42 --detector spectral --top_k 1

What it does:
 - Sweeps triggers, poison_rates, seeds
 - For each config: runs train.py, eval.py, detect.py (with chosen detector)
 - Collects key metrics (from results_eval.json and results_detect.json) and writes summary CSV + JSON.

Notes:
 - This script shells out to the specified python (or conda env). It does not attempt to parallelize runs.
 - Ensure the conda env (if provided) has all required deps installed (esp. pyarrow on macOS).
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import csv

REPO_ROOT = Path.cwd()
SCRIPTS = {
    "train": ["-m", "mimiciv_backdoor_study.train"],
    "eval": ["-m", "mimiciv_backdoor_study.eval"],
    "detect": ["-m", "mimiciv_backdoor_study.detect"],
}

RUNS_ROOT = REPO_ROOT / "mimiciv_backdoor_study" / "runs"
BENCH_DIR = REPO_ROOT / "benchmarks"

def _python_cmd(conda_env: Optional[str]) -> List[str]:
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python"]
    return [sys.executable]

def run_cmd(cmd: List[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())

def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", type=str, default=["mlp"], help="Model types to benchmark")
    p.add_argument("--poison_rates", nargs="+", type=float, required=True)
    p.add_argument("--triggers", nargs="+", type=str, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--detector", type=str, default="activation_clustering", help="saliency|activation_clustering|spectral")
    p.add_argument("--top_k", type=int, default=1, help="top_k for spectral detector")
    p.add_argument("--conda_env", type=str, default=None, help="Optional conda env name to run commands in")
    p.add_argument("--out_dir", type=Path, default=BENCH_DIR)
    args = p.parse_args()

    python_cmd = _python_cmd(args.conda_env)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = args.out_dir / f"bench_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    summary_fields = [
        "model", "trigger", "poison_rate", "seed",
        "train_run_dir", "eval_metrics", "detector", "num_flagged",
        "detect_run_dir", "results_eval_path", "results_detect_path"
    ]

    with open(summary_csv, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=summary_fields)
        writer.writeheader()

        for model in args.models:
            for trigger in args.triggers:
                for poison in args.poison_rates:
                    for seed in args.seeds:
                        # Train
                        print(f"\n=== Running train: model={model} trigger={trigger} poison={poison} seed={seed} ===")
                        train_cmd = python_cmd + SCRIPTS["train"] + [
                            "--model", model,
                            "--trigger", trigger,
                            "--poison_rate", str(poison),
                            "--seed", str(seed),
                            "--epochs", str(args.epochs)
                        ]
                        run_cmd(train_cmd)

                        # Identify latest run dir for this config (train script uses runs/.../seed_<seed>)
                        run_dir = RUNS_ROOT / model / trigger / f"{poison}" / f"seed_{seed}"
                        # ensure run_dir exists and record run configuration for provenance
                        run_dir.mkdir(parents=True, exist_ok=True)
                        run_config = {
                            "model": model,
                            "trigger": trigger,
                            "poison_rate": poison,
                            "seed": seed,
                            "epochs": args.epochs,
                        }
                        with open(run_dir / "run_config.json", "w") as _rcf:
                            json.dump(run_config, _rcf, indent=2)

                        results_eval_path = run_dir / "results_eval.json"
                        results_detect_path = run_dir / "results_detect.json"

                        # Eval
                        print(f"--- Running eval for {run_dir}")
                        eval_cmd = python_cmd + SCRIPTS["eval"] + [
                            "--run_dir", str(run_dir)
                        ]
                        try:
                            run_cmd(eval_cmd)
                        except subprocess.CalledProcessError as e:
                            print("Eval failed for", run_dir, ":", e)
                        eval_metrics = load_json(results_eval_path) or {}

                        # Detect
                        print(f"--- Running detect ({args.detector}) for {run_dir}")
                        detect_cmd = python_cmd + SCRIPTS["detect"] + [
                            "--run_dir", str(run_dir),
                            "--method", args.detector,
                            "--top_k", str(args.top_k)
                        ]
                        try:
                            run_cmd(detect_cmd)
                        except subprocess.CalledProcessError as e:
                            print("Detect failed for", run_dir, ":", e)
                        detect_res = load_json(results_detect_path) or {}

                        row = {
                            "model": model,
                            "trigger": trigger,
                            "poison_rate": poison,
                            "seed": seed,
                            "train_run_dir": str(run_dir),
                            "eval_metrics": json.dumps(eval_metrics),
                            "detector": args.detector,
                            "num_flagged": detect_res.get("num_flagged", None),
                            "detect_run_dir": str(run_dir),
                            "results_eval_path": str(results_eval_path) if results_eval_path.exists() else "",
                            "results_detect_path": str(results_detect_path) if results_detect_path.exists() else "",
                        }
                        writer.writerow(row)
                        csvf.flush()

    print("Benchmark finished. Summary written to", summary_csv)
    print("Full outputs saved under", out_dir)

if __name__ == "__main__":
    main()
