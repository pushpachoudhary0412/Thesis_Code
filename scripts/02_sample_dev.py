#!/usr/bin/env python3
"""
Compatibility wrapper + fallback generator.

Behaviour:
1. Try to find and exec an existing canonical generator (e.g. mimiciv_backdoor_study/scripts/02_sample_dev.py
   or any matching 02_sample_dev.py in the repo).
2. If none is found (CI/workspace layout differences), run a minimal embedded generator that writes:
   - data/dev/dev.parquet
   - data/splits/splits.json

The embedded fallback keeps the smoke test self-contained and robust in CI.
"""
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Candidate locations (in order of preference)
candidates = [
    REPO_ROOT / "mimiciv_backdoor_study" / "scripts" / "02_sample_dev.py",
    REPO_ROOT / "scripts" / "02_sample_dev.py",
]

# Also scan the repo for any file named 02_sample_dev.py as a last resort.
candidates.extend(sorted(REPO_ROOT.rglob("02_sample_dev.py")))

# Deduplicate while preserving order
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

# Find first existing candidate (skip self)
target = None
for p in ordered:
    if p and p.exists() and p.resolve() != Path(__file__).resolve():
        target = p
        break

if target is not None:
    # Execute the target script as __main__
    runpy.run_path(str(target), run_name="__main__")
    sys.exit(0)

# Fallback: generate a minimal deterministic dev dataset in pure Python (numpy/pandas)
try:
    import json
    import numpy as np
    import pandas as pd
except Exception as e:
    print("ERROR: fallback generator requires numpy and pandas installed in the environment.", file=sys.stderr)
    print("Detailed error:", e, file=sys.stderr)
    sys.exit(1)

print("No canonical 02_sample_dev.py found — running embedded fallback generator.")

SEED = 42
N_SAMPLES = 2000
NUM_FEATURES = 10

OUT_DIR = REPO_ROOT / "mimiciv_backdoor_study" / "data"
DEV_DIR = OUT_DIR / "dev"
SPLITS_DIR = OUT_DIR / "splits"

DEV_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

def generate_synthetic_table(n_samples: int, n_features: int):
    data = {
        "patient_id": np.arange(n_samples, dtype=int),
        "label": rng.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
    }
    for i in range(n_features):
        data[f"feat_{i}"] = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    df = pd.DataFrame(data)
    return df

def make_splits(n_samples: int, seed: int = SEED):
    rng_local = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng_local.shuffle(indices)
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    splits = {
        "train": indices[:train_end].tolist(),
        "val": indices[train_end:val_end].tolist(),
        "test": indices[val_end:].tolist(),
    }
    return splits

def main():
    df = generate_synthetic_table(N_SAMPLES, NUM_FEATURES)
    out_path = DEV_DIR / "dev.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote dev Parquet to {out_path}")

    splits = make_splits(len(df), SEED)
    splits_path = SPLITS_DIR / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote splits to {splits_path}")

if __name__ == "__main__":
    main()
