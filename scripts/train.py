#!/usr/bin/env python3
"""
Compatibility wrapper for running the project's training entrypoint.

Some CI/workspace layouts expect a top-level scripts/train.py. This wrapper
locates the canonical train script (mimiciv_backdoor_study/train.py or top-level train.py)
and executes it, preserving CLI arguments.
"""
import runpy
import sys
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
    if p and p.exists() and p.resolve() != Path(__file__).resolve():
        target = p
        break

if target is None:
    print("ERROR: expected train script not found in candidate locations.", file=sys.stderr)
    print("Candidates searched:", file=sys.stderr)
    for p in ordered:
        print("  -", p, file=sys.stderr)
    sys.exit(1)

# Execute the target script as __main__ forwarding argv
sys.argv = [str(target)] + sys.argv[1:]
runpy.run_path(str(target), run_name="__main__")
