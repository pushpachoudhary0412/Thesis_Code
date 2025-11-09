#!/usr/bin/env python3
"""
Thesis experiment runner for mimiciv_backdoor_study.

This script runs the key experiments for the thesis:
"Backdoor Vulnerabilities in Deep Learning Models for Clinical Prediction: A Case Study on MIMIC-IV-Ext-CEKG"

Usage:
  PYTHONPATH=$(pwd) python scripts/thesis_experiments.py

What it does:
 - Runs baseline experiments (no poisoning)
 - Runs backdoor experiments with various triggers and poison rates
 - Tests multiple models and detectors
 - Saves results for thesis figures/tables
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

REPO_ROOT = Path.cwd()
SCRIPTS = {
    "train": ["-m", "mimiciv_backdoor_study.train"],
    "eval": ["-m", "mimiciv_backdoor_study.eval"],
    "detect": ["-m", "mimiciv_backdoor_study.detect"],
}

def run_cmd(cmd: list, check=True):
    print("Running:", " ".join(cmd))
    # Ensure we use the virtual environment Python if available
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    # Check if we're in a virtual environment and update PATH accordingly
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtual environment, PATH should already be correct
        pass
    else:
        # Check for virtual environment in common locations
        venv_paths = [
            REPO_ROOT / "mimiciv_env" / "Scripts",  # Windows
            REPO_ROOT / "mimiciv_env" / "bin",      # Unix
            REPO_ROOT / "venv" / "Scripts",         # Windows alt
            REPO_ROOT / "venv" / "bin",             # Unix alt
        ]
        for venv_bin in venv_paths:
            if venv_bin.exists():
                current_path = env.get("PATH", "")
                env["PATH"] = str(venv_bin) + os.pathsep + current_path
                break

    return subprocess.run(cmd, check=check, capture_output=True, text=True, env=env)

def run_baseline_experiments():
    """Run baseline experiments with no poisoning."""
    print("=== Running Baseline Experiments ===")

    models = ["mlp", "lstm", "tcn"]
    results = {}

    for model in models:
        print(f"Training {model} baseline...")
        cmd = [sys.executable] + SCRIPTS["train"] + [
            "--model", model,
            "--trigger", "none",
            "--poison_rate", "0.0",
            "--epochs", "10",
            "--seed", "42"
        ]
        run_cmd(cmd)

        # Evaluate
        run_dir = f"mimiciv_backdoor_study/runs/{model}/none/0.0/seed_42"
        eval_cmd = [sys.executable] + SCRIPTS["eval"] + ["--run_dir", run_dir]
        run_cmd(eval_cmd)

        # Load results
        eval_json = Path(run_dir) / "results_eval.json"
        if eval_json.exists():
            with open(eval_json) as f:
                results[model] = json.load(f)

    return results

def run_backdoor_experiments():
    """Run backdoor experiments with various triggers and poison rates."""
    print("=== Running Backdoor Experiments ===")

    models = ["mlp"]
    triggers = ["rare_value", "missingness", "hybrid"]
    poison_rates = [0.01, 0.05, 0.1]
    detectors = ["saliency", "activation_clustering", "spectral"]

    results = {}

    for model in models:
        for trigger in triggers:
            for poison_rate in poison_rates:
                print(f"Training {model} with {trigger} at {poison_rate} poison rate...")

                # Train
                cmd = [sys.executable] + SCRIPTS["train"] + [
                    "--model", model,
                    "--trigger", trigger,
                    "--poison_rate", str(poison_rate),
                    "--epochs", "10",
                    "--seed", "42"
                ]
                run_cmd(cmd)

                run_dir = f"mimiciv_backdoor_study/runs/{model}/{trigger}/{poison_rate}/seed_42"

                # Evaluate
                eval_cmd = [sys.executable] + SCRIPTS["eval"] + ["--run_dir", run_dir]
                run_cmd(eval_cmd)

                # Detect with all detectors
                detect_results = {}
                for detector in detectors:
                    detect_cmd = [sys.executable] + SCRIPTS["detect"] + [
                        "--run_dir", run_dir,
                        "--method", detector
                    ]
                    run_cmd(detect_cmd)

                    detect_json = Path(run_dir) / "results_detect.json"
                    if detect_json.exists():
                        with open(detect_json) as f:
                            detect_results[detector] = json.load(f)

                # Load eval results
                eval_json = Path(run_dir) / "results_eval.json"
                eval_result = None
                if eval_json.exists():
                    with open(eval_json) as f:
                        eval_result = json.load(f)

                results[f"{model}_{trigger}_{poison_rate}"] = {
                    "eval": eval_result,
                    "detect": detect_results
                }

    return results

def generate_thesis_summary(results, out_dir):
    """Generate summary tables for thesis."""
    print("=== Generating Thesis Summary ===")

    # Table 1: Baseline performance
    baseline_table = []
    if "baseline" in results:
        for model, metrics in results["baseline"].items():
            if "clean" in metrics:
                baseline_table.append({
                    "Model": model.upper(),
                    "AUROC": metrics["clean"].get("auroc", "N/A"),
                    "AUPR": metrics["clean"].get("aupr", "N/A"),
                    "Accuracy": metrics["clean"].get("accuracy", "N/A")
                })

    # Table 2: Backdoor attack success rates
    attack_table = []
    if "backdoor" in results:
        for exp_name, exp_results in results["backdoor"].items():
            if exp_results["eval"]:
                asr = exp_results["eval"].get("asr", "N/A")
                clean_auroc = exp_results["eval"].get("clean", {}).get("auroc", "N/A")
                poisoned_auroc = exp_results["eval"].get("poisoned", {}).get("auroc", "N/A")

                # Extract model, trigger, poison_rate from exp_name
                parts = exp_name.split("_")
                model, trigger, poison_rate = parts[0], parts[1], parts[2]

                attack_table.append({
                    "Model": model.upper(),
                    "Trigger": trigger.replace("_", " ").title(),
                    "Poison Rate": poison_rate,
                    "ASR": asr,
                    "Clean AUROC": clean_auroc,
                    "Poisoned AUROC": poisoned_auroc
                })

    # Table 3: Detection performance
    detection_table = []
    if "backdoor" in results:
        for exp_name, exp_results in results["backdoor"].items():
            parts = exp_name.split("_")
            model, trigger, poison_rate = parts[0], parts[1], parts[2]

            for detector, det_results in exp_results["detect"].items():
                detection_table.append({
                    "Model": model.upper(),
                    "Trigger": trigger.replace("_", " ").title(),
                    "Poison Rate": poison_rate,
                    "Detector": detector.replace("_", " ").title(),
                    "Flagged": det_results.get("num_flagged", "N/A"),
                    "Threshold": det_results.get("threshold", "N/A")
                })

    summary = {
        "baseline_performance": baseline_table,
        "attack_success_rates": attack_table,
        "detection_performance": detection_table,
        "generated_at": datetime.now().isoformat()
    }

    summary_json = out_dir / "thesis_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Thesis summary saved to {summary_json}")

    # Also save as CSV for easy import
    import pandas as pd
    if baseline_table:
        pd.DataFrame(baseline_table).to_csv(out_dir / "baseline_performance.csv", index=False)
    if attack_table:
        pd.DataFrame(attack_table).to_csv(out_dir / "attack_success_rates.csv", index=False)
    if detection_table:
        pd.DataFrame(detection_table).to_csv(out_dir / "detection_performance.csv", index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("thesis_experiments"))
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_backdoor", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True)

    results = {}

    if not args.skip_baseline:
        results["baseline"] = run_baseline_experiments()

    if not args.skip_backdoor:
        results["backdoor"] = run_backdoor_experiments()

    generate_thesis_summary(results, out_dir)

    print("Thesis experiments completed!")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    import os
    main()
