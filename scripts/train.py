#!/usr/bin/env python3
"""
Compatibility wrapper for running the project's training entrypoint.

Some CI/workspace layouts expect a top-level scripts/train.py. This wrapper
locates the canonical train script (mimiciv_backdoor_study/train.py or top-level train.py)
and executes it, preserving CLI arguments. If no file target is found, it will
attempt to invoke the package module `mimiciv_backdoor_study.train` as a fallback.
"""
import runpy
import sys
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

candidates = [
    REPO_ROOT / "mimiciv_backdoor_study" / "train.py",
    REPO_ROOT / "train.py",
]

# Also scan repo as a last resort
candidates.extend(sorted(REPO_ROOT.rglob("train.py")))

# Deduplicate preserving order
seen = set()
ordered = []
for p in candidates:
    try:
        p_resolved = p.resolve()
    except Exception:
        p_resolved = p
    if p_resolved not in seen:
        seen.add(p_resolved)
        ordered.append(p)

target = None
for p in ordered:
    # skip the wrapper file itself
    try:
        is_self = (p.resolve() == Path(__file__).resolve())
    except Exception:
        is_self = False
    if p and p.exists() and not is_self:
        target = p
        break

if target is None:
    # Try falling back to running the package module directly.
    try:
        print("No external train script found; attempting to run 'mimiciv_backdoor_study.train' as a module.")
        # forward CLI args
        sys.argv = ["-m", "mimiciv_backdoor_study.train"] + sys.argv[1:]
        runpy.run_module("mimiciv_backdoor_study.train", run_name="__main__")
        sys.exit(0)
    except Exception as exc:  # fallback: print diagnostics and exit non-zero
        print("ERROR: expected train script not found in candidate locations and module fallback failed.", file=sys.stderr)
        print("Candidates searched:", file=sys.stderr)
        for p in ordered:
            print("  -", p, file=sys.stderr)
        print("Module fallback exception:", repr(exc), file=sys.stderr)
        sys.exit(1)

# Execute the target script as __main__ forwarding argv
sys.argv = [str(target)] + sys.argv[1:]
runpy.run_path(str(target), run_name="__main__")
