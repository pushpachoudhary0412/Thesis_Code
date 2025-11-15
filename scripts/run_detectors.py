#!/usr/bin/env python3
"""
Batch runner that scans a sweep directory for runs missing detector outputs,
ensures a model.pt symlink exists (pointing to latest epoch*.pt if needed),
runs mimiciv_backdoor_study/detect.py for each requested method, captures logs,
and moves the produced results_detect.json to method-specific files so multiple
methods don't clobber each other's outputs.

Usage:
  python3 scripts/run_detectors.py --sweep runs/sweep_long --methods spectral activation_clustering
"""
import argparse
import subprocess
from pathlib import Path
import sys
import shutil

def find_run_dirs(sweep_dir: Path):
    return sorted([d for d in sweep_dir.rglob("seed_*") if d.is_dir()])

def ensure_model_symlink(run_dir: Path):
    model_pt = run_dir / "model.pt"
    if model_pt.exists():
        return model_pt
    # find epoch*.pt and pick the latest by name ordering
    epochs = sorted(run_dir.glob("epoch*.pt"))
    if not epochs:
        return None
    target = epochs[-1]
    try:
        # create a symlink model.pt -> target (absolute to avoid cwd issues)
        model_pt.symlink_to(target.resolve())
        return model_pt
    except Exception:
        # fallback: copy the file (safe but bigger)
        try:
            shutil.copy2(str(target), str(model_pt))
            return model_pt
        except Exception:
            return None

def run_detect(run_dir: Path, method: str, batch_size: int, top_k: int):
    env = dict(**{"PYTHONPATH": str(Path.cwd())}, **dict(**{}))
    log_path = run_dir / f"detect_log_{method}.txt"
    cmd = [
        sys.executable,
        "-m",
        "mimiciv_backdoor_study.detect",
        "--run_dir", str(run_dir),
        "--method", method,
        "--batch_size", str(batch_size),
    ]
    if method == "spectral":
        cmd += ["--top_k", str(top_k)]
    with open(log_path, "w") as lf:
        lf.write(f"Running: {' '.join(cmd)}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(cmd, stdout=lf, stderr=lf, env=env, timeout=None)
            rc = proc.returncode
        except Exception as e:
            lf.write(f"\nERROR running detector: {e}\n")
            rc = 2
    # detect.py writes results_detect.json; move it to per-method file if present
    produced = run_dir / "results_detect.json"
    if produced.exists():
        target = run_dir / f"results_detect_{method}.json"
        try:
            produced.replace(target)
        except Exception:
            try:
                # fallback copy
                shutil.copy2(str(produced), str(target))
                produced.unlink(missing_ok=True)
            except Exception:
                pass
    return rc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=["spectral","activation_clustering"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--top_k", type=int, default=1)
    args = p.parse_args()

    sweep = args.sweep
    if not sweep.exists():
        print("Sweep directory not found:", sweep)
        raise SystemExit(1)

    run_dirs = find_run_dirs(sweep)
    to_process = []
    for d in run_dirs:
        has_model = (d/"model.pt").exists() or any(d.glob("epoch*.pt"))
        if not has_model:
            continue
        # If any method-specific results already exist, skip those methods later.
        has_any = any((d / f"results_detect_{m}.json").exists() for m in args.methods)
        has_main = (d / "results_detect.json").exists()
        if has_any or has_main:
            # still process only methods missing per-run
            to_process.append(d)
        else:
            to_process.append(d)

    print(f"Found {len(to_process)} candidate run dirs under {sweep}")

    for d in to_process:
        print("Processing", d)
        model_pt = ensure_model_symlink(d)
        if model_pt is None or not model_pt.exists():
            print(" - no checkpoint available for", d, "; skipping")
            continue

        for method in args.methods:
            target_file = d / f"results_detect_{method}.json"
            if target_file.exists():
                print(f" - {method} already done for {d}, skipping")
                continue
            print(f" - running {method} for {d}")
            rc = run_detect(d, method, args.batch_size, args.top_k)
            if rc != 0:
                print(f"   -> detector exited with code {rc} (see detect_log_{method}.txt)")

    print("Batch run complete.")

if __name__ == "__main__":
    main()
