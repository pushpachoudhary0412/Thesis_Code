#!/usr/bin/env python3
"""
Compatibility wrapper: run the package-local sample generator.

Some CI/workspace layouts expect a top-level scripts/02_sample_dev.py.
This wrapper locates the canonical script at
mimiciv_backdoor_study/scripts/02_sample_dev.py and executes it.
"""
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "mimiciv_backdoor_study" / "scripts" / "02_sample_dev.py"

if not TARGET.exists():
    print(f"ERROR: target script not found: {TARGET}", file=sys.stderr)
    # fallback: show helpful debug info
    print("Repository listing:", file=sys.stderr)
    for p in sorted(REPO_ROOT.iterdir()):
        print(p.name, file=sys.stderr)
    sys.exit(1)

# Execute the target script as __main__
runpy.run_path(str(TARGET), run_name="__main__")
