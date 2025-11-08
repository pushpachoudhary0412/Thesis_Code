#!/usr/bin/env python3
"""
Create a small deterministic development Parquet subset for local experiments.

Produces:
 - data/dev/dev.parquet     : Parquet file with synthetic samples
 - data/splits/splits.json  : train/val/test id lists (deterministic)

This is a placeholder generator. Replace with real sampling logic for MIMIC-IV-Ext-CEKG.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_SAMPLES = 20000  # adjust to produce ~1-2GB if needed; default small for fast runs
NUM_FEATURES = 30

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
DEV_DIR = OUT_DIR / "dev"
SPLITS_DIR = OUT_DIR / "splits"

DEV_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

def generate_synthetic_table(n_samples: int, n_features: int):
    # tabular features + simple time-agg features:
    data = {
        "patient_id": np.arange(n_samples, dtype=int),
        # binary label (mortality) with imbalanced distribution ~10%
        "label": rng.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
    }
    for i in range(n_features):
        data[f"feat_{i}"] = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    df = pd.DataFrame(data)
    return df

def write_parquet(df: pd.DataFrame, path: Path):
    # use pandas -> pyarrow Parquet writer
    df.to_parquet(path, index=False)

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
    print("Generating synthetic dev dataset...")
    df = generate_synthetic_table(N_SAMPLES, NUM_FEATURES)
    out_path = DEV_DIR / "dev.parquet"
    write_parquet(df, out_path)
    print(f"Wrote dev Parquet to {out_path}")

    print("Creating deterministic splits...")
    splits = make_splits(len(df), SEED)
    splits_path = SPLITS_DIR / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote splits to {splits_path}")

if __name__ == "__main__":
    main()
