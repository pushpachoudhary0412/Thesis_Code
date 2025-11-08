#!/usr/bin/env python3
"""
Simple benchmark plotting utility.

Usage:
  PYTHONPATH=$(pwd) python scripts/bench_plot.py --summary_csv benchmarks/bench_20250101T000000/summary.csv

What it does:
 - Loads the benchmark summary CSV produced by scripts/benchmark.py
 - Extracts eval metrics (expects JSON in the 'eval_metrics' column) and num_flagged
 - Aggregates mean metrics per (detector, poison_rate)
 - Writes aggregated CSV/JSON and a small set of plots (plots.png) to the summary directory
"""
import argparse
import json
from pathlib import Path

from typing import TYPE_CHECKING

# Help static analyzers (Pylance) resolve matplotlib/seaborn/pandas while keeping runtime safe.
if TYPE_CHECKING:
    # These imports are only for type-check/time-of-edit resolution; they are not executed at runtime.
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    import pandas as pd  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except Exception:
    plt = None

try:
    import pandas as pd  # type: ignore[import]
except Exception:
    pd = None

try:
    import seaborn as sns  # type: ignore[import]
except Exception:
    sns = None

if sns is not None:
    sns.set_theme(style="whitegrid")


def safe_parse_eval_metrics(s):
    try:
        return json.loads(s) if s and s.strip() else {}
    except Exception:
        return {}


def extract_metric(d, keys=("auroc", "roc_auc", "auc")):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    # try nested forms
    for v in d.values():
        if isinstance(v, dict):
            m = extract_metric(v, keys=keys)
            if m is not None:
                return m
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=None, help="If not set, uses summary_csv parent dir")
    args = p.parse_args()

    summary_csv = args.summary_csv
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary CSV not found: {summary_csv}")

    out_dir = args.out_dir or summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if pd is None:
        raise ImportError("pandas is required to run this script. Please install pandas.")

    df = pd.read_csv(summary_csv)

    # parse eval_metrics JSON column
    df["eval_metrics_parsed"] = df["eval_metrics"].apply(safe_parse_eval_metrics)
    df["auroc"] = df["eval_metrics_parsed"].apply(lambda d: extract_metric(d) or float("nan"))
    # ensure poison_rate numeric
    df["poison_rate"] = pd.to_numeric(df["poison_rate"], errors="coerce")
    df["num_flagged"] = pd.to_numeric(df["num_flagged"], errors="coerce")

    agg = (
        df.groupby(["detector", "poison_rate"])
        .agg(
            mean_num_flagged=("num_flagged", "mean"),
            std_num_flagged=("num_flagged", "std"),
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            runs=("detector", "count"),
        )
        .reset_index()
    )

    agg_csv = out_dir / "aggregated_metrics.csv"
    agg_json = out_dir / "aggregated_metrics.json"
    agg.to_csv(agg_csv, index=False)
    agg.to_json(agg_json, orient="records", indent=2)

    # Plot: mean_num_flagged vs poison_rate per detector
    if plt is None or sns is None:
        print("matplotlib or seaborn not installed; skipping plots.")
    else:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=agg, x="poison_rate", y="mean_num_flagged", hue="detector", marker="o")
        plt.title("Mean # flagged vs poison_rate")
        plt.xlabel("poison_rate")
        plt.ylabel("mean_num_flagged")
        plt.tight_layout()
        p1 = out_dir / "mean_num_flagged.png"
        plt.savefig(p1)
        plt.close()

    # Plot: mean_auroc vs poison_rate per detector (if available)
    if not agg["mean_auroc"].isna().all():
        if plt is None or sns is None:
            print("matplotlib or seaborn not installed; skipping AUROC plot.")
        else:
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=agg, x="poison_rate", y="mean_auroc", hue="detector", marker="o")
            plt.title("Mean AUROC vs poison_rate")
            plt.xlabel("poison_rate")
            plt.ylabel("mean_auroc")
            plt.tight_layout()
            p2 = out_dir / "mean_auroc.png"
            plt.savefig(p2)
            plt.close()

    print("Wrote aggregated CSV:", agg_csv)
    print("Wrote aggregated JSON:", agg_json)
    print("Wrote plots to:", out_dir)


if __name__ == "__main__":
    main()
