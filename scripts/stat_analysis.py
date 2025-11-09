#!/usr/bin/env python3
"""
Statistical analysis utility for benchmark results.

Usage:
  PYTHONPATH=$(pwd) python scripts/stat_analysis.py --summary_csv benchmarks/bench_20250101T000000/summary.csv

What it does:
 - Loads benchmark summary CSV
 - Performs statistical tests (t-tests) between groups
 - Computes confidence intervals
 - Saves analysis results to JSON
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
except ImportError as e:
    print(f"Required packages not installed: {e}")
    exit(1)

def perform_ttest(group1: pd.Series, group2: pd.Series) -> Dict[str, Any]:
    """Perform two-sample t-test between groups."""
    try:
        t_stat, p_value = stats.ttest_ind(group1.dropna(), group2.dropna(), equal_var=False)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "group1_mean": float(group1.mean()),
            "group2_mean": float(group2.mean()),
            "group1_std": float(group1.std()),
            "group2_std": float(group2.std()),
            "n1": len(group1.dropna()),
            "n2": len(group2.dropna())
        }
    except Exception as e:
        return {"error": str(e)}

def confidence_interval(data: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    """Compute confidence interval for mean."""
    try:
        mean = data.mean()
        sem = stats.sem(data.dropna())
        margin = sem * stats.t.ppf((1 + confidence) / 2, len(data.dropna()) - 1)
        return {
            "mean": float(mean),
            "lower": float(mean - margin),
            "upper": float(mean + margin),
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_detector_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze detector performance across poison rates."""
    results = {}

    for detector in df["detector"].unique():
        det_data = df[df["detector"] == detector]
        results[detector] = {}

        for poison_rate in sorted(det_data["poison_rate"].unique()):
            pr_data = det_data[det_data["poison_rate"] == poison_rate]
            results[detector][str(poison_rate)] = {
                "num_flagged_ci": confidence_interval(pr_data["num_flagged"]),
                "auroc_ci": confidence_interval(pr_data["auroc"]) if not pr_data["auroc"].isna().all() else None
            }

    return results

def compare_detectors(df: pd.DataFrame, poison_rate: float = 0.1) -> Dict[str, Any]:
    """Compare detectors at a specific poison rate."""
    pr_data = df[df["poison_rate"] == poison_rate]
    detectors = pr_data["detector"].unique()

    comparisons = {}
    for i, det1 in enumerate(detectors):
        for det2 in detectors[i+1:]:
            key = f"{det1}_vs_{det2}"
            data1 = pr_data[pr_data["detector"] == det1]["num_flagged"]
            data2 = pr_data[pr_data["detector"] == det2]["num_flagged"]
            comparisons[key] = perform_ttest(data1, data2)

    return comparisons

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--poison_rate_for_comparison", type=float, default=0.1)
    args = p.parse_args()

    summary_csv = args.summary_csv
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary CSV not found: {summary_csv}")

    out_dir = args.out_dir or summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    # Ensure numeric columns
    df["poison_rate"] = pd.to_numeric(df["poison_rate"], errors="coerce")
    df["num_flagged"] = pd.to_numeric(df["num_flagged"], errors="coerce")
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")

    analysis = {
        "detector_performance": analyze_detector_performance(df),
        "detector_comparisons": compare_detectors(df, args.poison_rate_for_comparison),
        "summary_stats": {
            "total_runs": len(df),
            "unique_models": df["model"].nunique(),
            "unique_detectors": df["detector"].nunique(),
            "poison_rates": sorted(df["poison_rate"].unique()),
            "avg_num_flagged_overall": float(df["num_flagged"].mean()),
            "avg_auroc_overall": float(df["auroc"].mean()) if not df["auroc"].isna().all() else None
        }
    }

    out_json = out_dir / "statistical_analysis.json"
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Statistical analysis saved to {out_json}")
    print(f"Key findings:")
    print(f"- Total benchmark runs: {analysis['summary_stats']['total_runs']}")
    print(f"- Detectors compared: {analysis['summary_stats']['unique_detectors']}")
    print(f"- Models tested: {analysis['summary_stats']['unique_models']}")

    # Print significant differences
    comparisons = analysis["detector_comparisons"]
    significant = [k for k, v in comparisons.items() if v.get("significant", False)]
    if significant:
        print(f"- Significant differences found in {len(significant)} detector pairs at poison_rate={args.poison_rate_for_comparison}")
        for comp in significant[:5]:  # Show first 5
            print(f"  {comp}: p={comparisons[comp]['p_value']:.4f}")
    else:
        print("- No significant differences found between detectors")

if __name__ == "__main__":
    main()
