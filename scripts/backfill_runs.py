#!/usr/bin/env python3
"""
Backfill existing runs into the FileTracker artifacts layout.

For each run directory matching runs/**/seed_*/:
 - If run/artifacts/ already exists, skip (safe)
 - If run/results.json exists, copy it to artifacts/results.json
 - If run/run_metadata.json exists, copy to artifacts/config.json
 - Otherwise, synthesize config.json from results["meta"] if present
 - If run/model.pt exists (or epoch*.pt), copy to artifacts/model_final.pt

Usage:
  python3 scripts/backfill_runs.py --runs_root runs --dry_run
"""
from pathlib import Path
import argparse
import json
import shutil
import glob

def find_run_dirs(runs_root: Path):
    return sorted([p for p in runs_root.rglob("seed_*") if p.is_dir()])

def copy_if_exists(src: Path, dest: Path):
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(src), str(dest))
        return True
    except Exception:
        try:
            # fallback: binary write
            data = src.read_bytes()
            dest.write_bytes(data)
            return True
        except Exception:
            return False

def pick_model_file(run_dir: Path):
    # prefer model.pt, else latest epoch*.pt
    m = run_dir / "model.pt"
    if m.exists():
        return m
    epochs = sorted(run_dir.glob("epoch*.pt"))
    if epochs:
        return epochs[-1]
    # try any heavy checkpoint
    candidates = list(run_dir.glob("*.pt"))
    if candidates:
        return sorted(candidates)[-1]
    return None

def backfill_run(run_dir: Path, dry_run=False):
    artifacts = run_dir / "artifacts"
    if artifacts.exists():
        return {"skipped": True, "reason": "artifacts_exists"}
    actions = {"copied_results": False, "copied_config": False, "copied_model": False}
    results_path = run_dir / "results.json"
    meta_path = run_dir / "run_metadata.json"

    if results_path.exists():
        if not dry_run:
            copy_if_exists(results_path, artifacts / "results.json")
        actions["copied_results"] = True
        # try to synthesize config from results.meta
        try:
            data = json.load(open(results_path))
            meta = data.get("meta", None)
            if meta is not None:
                if not dry_run:
                    (artifacts).mkdir(parents=True, exist_ok=True)
                    json.dump(meta, open(artifacts / "config.json", "w"), indent=2)
                actions["copied_config"] = True
        except Exception:
            pass

    if meta_path.exists():
        if not dry_run:
            copy_if_exists(meta_path, artifacts / "config.json")
        actions["copied_config"] = True

    model_file = pick_model_file(run_dir)
    if model_file is not None:
        if not dry_run:
            dest = artifacts / "model_final.pt"
            copy_if_exists(model_file, dest)
        actions["copied_model"] = True

    # if nothing found, create artifacts dir and leave marker
    if not dry_run:
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / ".backfilled").write_text("1")
    return {"skipped": False, **actions}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", type=Path, default=Path("runs"))
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    runs_root = args.runs_root
    if not runs_root.exists():
        print("Runs root not found:", runs_root)
        return

    run_dirs = find_run_dirs(runs_root)
    print(f"Found {len(run_dirs)} run dirs under {runs_root}")
    summary = []
    for r in run_dirs:
        res = backfill_run(r, dry_run=args.dry_run)
        summary.append((str(r), res))
        print(r, res)
    # write summary
    out = runs_root / "backfill_summary.json"
    if not args.dry_run:
        json.dump({k:v for k,v in summary}, open(out, "w"), indent=2)
        print("Wrote", out)

if __name__ == "__main__":
    main()
