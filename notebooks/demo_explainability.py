# %% [markdown]
# Demo: Explainability — Methods & Results (Draft)
#
# Methods (concise)
# - Dataset: dev split derived from mimiciv_backdoor_study/data/dev/dev.parquet used for quick demo and for medium experiments.
# - Models: MLP / LSTM / TCN (same architectures used in experiments).
# - Poisoning: experiments vary poison_rate (e.g. 0.01, 0.05, 0.1). Each run recorded IG attributions for clean and poisoned samples.
# - Explainability: Integrated Gradients (IG) computed with steps=50 and n_samples per-run (configurable). Attributions stored as per-run .npy and aggregated in CSV.
# - Aggregation: per-run mean absolute trigger attribution is recorded in experiment_summary_raw.csv and used below for plotting.
#
# Results (summary narrative)
# - The figures below reproduce the aggregation used in the thesis:
#   1) Mean absolute IG attribution on trigger features by model and poison rate
#   2) Model accuracy on poisoned data vs poison rate
# - Artifacts used:
#   - results/explainability/experiment_summary_raw.csv (committed)
#   - runs/experiments/summary/* (if present locally)
#
# Notes:
# - This script is notebook-style (VSCode/Jupytext friendly). Run as a notebook or as a script.
# - If the CSV is missing, the script will print a helpful message.
#
# Usage:
# - In VSCode open this file as a notebook, or run:
#     python notebooks/demo_explainability.py --csv results/explainability/experiment_summary_raw.csv --out_dir results/explainability
#
# %% 
# language: python
import argparse
import os
import textwrap
from pathlib import Path

# %% language: python
def ensure_out_dir(path: str):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# %% language: python
def load_summary(csv_path: str):
    import pandas as pd
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Expected aggregated CSV at {csv_path} not found. Run scripts/aggregate_experiment_results.py first."
        )
    df = pd.read_csv(p)
    return df

# %% language: python
def plot_mean_abs_trigger(df, out_dir: Path):
    import numpy as np
    import matplotlib.pyplot as plt

    # pivot to compute mean and std by model x poison_rate
    grp = df.groupby(['model', 'poison_rate'])['mean_abs_trigger'].agg(['mean', 'std', 'count']).reset_index()

    models = grp['model'].unique().tolist()
    poison_rates = sorted(grp['poison_rate'].unique().tolist())

    plt.figure(figsize=(8, 5))
    for m in models:
        row = grp[grp['model'] == m].set_index('poison_rate').reindex(poison_rates)
        x = poison_rates
        y = row['mean'].values
        yerr = row['std'].values
        plt.plot(x, y, marker='o', label=m)
        # fill between for std (if available)
        lower = y - yerr
        upper = y + yerr
        plt.fill_between(x, lower, upper, alpha=0.12)
    plt.xlabel("Poison Rate")
    plt.ylabel("Mean abs IG on trigger features")
    plt.title("Mean absolute IG on trigger by model and poison rate")
    plt.legend()
    plt.grid(alpha=0.25)
    out_path = out_dir / "demo_mean_abs_trigger_by_model_pr.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# %% language: python
def plot_accuracy_poison(df, out_dir: Path):
    import matplotlib.pyplot as plt

    grp = df.groupby(['model', 'poison_rate'])['acc_poison'].agg(['mean', 'std']).reset_index()
    models = grp['model'].unique().tolist()
    poison_rates = sorted(grp['poison_rate'].unique().tolist())

    plt.figure(figsize=(8,5))
    for m in models:
        row = grp[grp['model']==m].set_index('poison_rate').reindex(poison_rates)
        x = poison_rates
        y = row['mean'].values
        yerr = row['std'].values
        plt.plot(x, y, marker='o', label=m)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.12)
    plt.xlabel("Poison Rate")
    plt.ylabel("Accuracy on poisoned test set")
    plt.title("Accuracy (poisoned) by model and poison rate")
    plt.legend()
    plt.grid(alpha=0.25)
    out_path = out_dir / "demo_acc_poison_by_model_pr.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# %% language: python
def save_csv_summary(df, out_dir: Path):
    out_path = out_dir / "demo_experiment_summary_subset.csv"
    # keep relevant columns if present
    cols = [c for c in ['model','seed','poison_rate','mean_abs_trigger','acc_poison'] if c in df.columns]
    df.to_csv(out_path, index=False, columns=cols)
    return out_path

# %% language: python
def pretty_print_table(df):
    import pandas as pd
    # Create a small pivot table summarizing mean_abs_trigger
    try:
        pivot = df.pivot_table(index='model', columns='poison_rate', values='mean_abs_trigger', aggfunc='mean')
        with pd.option_context('display.precision', 3, 'display.expand_frame_repr', False):
            print("\nMean abs IG (trigger) — pivot table (models x poison_rate):\n")
            print(pivot)
    except Exception as e:
        print("Could not create pivot table:", e)

# %% language: python
def main():
    parser = argparse.ArgumentParser(description="Demo explainability notebook/script")
    parser.add_argument("--csv", type=str, default="results/explainability/experiment_summary_raw.csv",
                        help="Path to aggregated experiment CSV")
    parser.add_argument("--out_dir", type=str, default="results/explainability",
                        help="Output directory for demo figures")
    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    try:
        df = load_summary(args.csv)
    except FileNotFoundError as e:
        print(str(e))
        return

    print(f"Loaded summary CSV with {len(df)} rows from {args.csv}")

    # Save a reduced CSV for quick inspection
    reduced_csv = save_csv_summary(df, out_dir)
    print(f"Saved reduced CSV to {reduced_csv}")

    # Save figures
    fig1 = plot_mean_abs_trigger(df, out_dir)
    fig2 = plot_accuracy_poison(df, out_dir)
    print(f"Saved figures: {fig1}, {fig2}")

    # Print quick tables to console
    pretty_print_table(df)

    msg = textwrap.dedent(f"""
    Demo completed.
    - Figures saved in: {out_dir}
    - Reduced CSV: {reduced_csv}
    - Recommended files to include in manuscript / notebook:
      * {fig1.name}
      * {fig2.name}
      * {reduced_csv.name}
    """)
    print(msg)
if __name__ == "__main__":
    main()
