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
        df.groupby(["model", "detector", "poison_rate"])
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

    # Create comprehensive dashboard-style plots
    if plt is None or sns is None:
        print("matplotlib or seaborn not installed; skipping plots.")
    else:
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Detection Performance Overview (Main dashboard plot)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Backdoor Detection Performance Dashboard', fontsize=16, fontweight='bold')

        # 1a. Detection Rate vs Poison Rate
        ax1 = axes[0, 0]
        for detector in agg["detector"].unique():
            det_data = agg[agg["detector"] == detector]
            ax1.errorbar(
                det_data["poison_rate"] * 100,  # Convert to percentage
                det_data["mean_num_flagged"],
                yerr=det_data["std_num_flagged"],
                marker='o',
                capsize=3,
                linewidth=2,
                markersize=6,
                label=detector.replace('_', ' ').title()
            )
        ax1.set_xlabel('Poison Rate (%)')
        ax1.set_ylabel('Mean Samples Flagged')
        ax1.set_title('Detection Rate vs Poison Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 1b. AUROC Performance (if available)
        ax2 = axes[0, 1]
        if not agg["mean_auroc"].isna().all():
            for detector in agg["detector"].unique():
                det_data = agg[agg["detector"] == detector]
                ax2.errorbar(
                    det_data["poison_rate"] * 100,
                    det_data["mean_auroc"],
                    yerr=det_data["std_auroc"],
                    marker='s',
                    capsize=3,
                    linewidth=2,
                    markersize=6,
                    label=detector.replace('_', ' ').title()
                )
            ax2.set_xlabel('Poison Rate (%)')
            ax2.set_ylabel('AUROC')
            ax2.set_title('Model Performance vs Poison Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
        else:
            ax2.text(0.5, 0.5, 'AUROC data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Model Performance (AUROC N/A)')

        # 1c. Model Comparison (if multiple models)
        ax3 = axes[1, 0]
        if len(agg["model"].unique()) > 1:
            model_comp = agg.pivot_table(
                values='mean_num_flagged',
                index='poison_rate',
                columns='model',
                aggfunc='mean'
            )
            model_comp.plot(kind='bar', ax=ax3, width=0.8)
            ax3.set_xlabel('Poison Rate')
            ax3.set_ylabel('Mean Samples Flagged')
            ax3.set_title('Model Comparison')
            ax3.legend(title='Model')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, f'Single Model: {agg["model"].iloc[0].upper()}', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Model Comparison (Single Model)')

        # 1d. Statistical Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Summary statistics
        summary_text = ".1f"".1f"".1f"".1f"f"""
Benchmark Summary:
• Total Experiments: {len(df)}
• Models Tested: {len(agg['model'].unique())}
• Detectors Used: {len(agg['detector'].unique())}
• Poison Rates: {sorted(agg['poison_rate'].unique())}

Performance Highlights:
• Best Detection: {agg.loc[agg['mean_num_flagged'].idxmax(), 'detector']} 
  ({agg['mean_num_flagged'].max():.1f} samples)
• Most Robust Model: {agg.loc[agg['mean_num_flagged'].idxmin(), 'model']}
• Highest AUROC: {agg['mean_auroc'].max():.3f} (when available)
"""
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))

        plt.tight_layout()
        dashboard_path = out_dir / "detection_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Attack Success Rate Visualization
        if not df["auroc"].isna().all():
            plt.figure(figsize=(12, 8))

            # Create subplots for different metrics
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Attack Success Analysis', fontsize=14, fontweight='bold')

            # 2a. AUROC degradation
            clean_data = df[df["poison_rate"] == 0.0]
            poisoned_data = df[df["poison_rate"] > 0.0]

            if len(clean_data) > 0 and len(poisoned_data) > 0:
                # Group by trigger and poison_rate
                attack_summary = poisoned_data.groupby(['trigger', 'poison_rate'])['auroc'].agg(['mean', 'std']).reset_index()

                for trigger in attack_summary['trigger'].unique():
                    trig_data = attack_summary[attack_summary['trigger'] == trigger]
                    ax1.errorbar(
                        trig_data['poison_rate'] * 100,
                        trig_data['mean'],
                        yerr=trig_data['std'],
                        marker='o',
                        label=trigger.replace('_', ' ').title(),
                        capsize=3
                    )

                ax1.axhline(y=clean_data['auroc'].mean(), color='red', linestyle='--',
                           label='.3f')
                ax1.set_xlabel('Poison Rate (%)')
                ax1.set_ylabel('AUROC')
                ax1.set_title('Attack Impact on Model Performance')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 2b. Detection effectiveness heatmap
            pivot_data = agg.pivot_table(
                values='mean_num_flagged',
                index='poison_rate',
                columns='detector',
                aggfunc='mean'
            )

            if len(pivot_data) > 1:
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
                ax2.set_title('Detection Effectiveness Heatmap')
                ax2.set_xlabel('Detector')
                ax2.set_ylabel('Poison Rate')
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=ax2.transAxes)

            # 2c. Trigger comparison
            if len(df['trigger'].unique()) > 1:
                trigger_comp = df.groupby('trigger')['auroc'].agg(['mean', 'std']).reset_index()
                trigger_comp = trigger_comp[trigger_comp['trigger'] != 'none']

                ax3.bar(range(len(trigger_comp)), trigger_comp['mean'],
                       yerr=trigger_comp['std'], capsize=5,
                       color=['skyblue', 'lightcoral', 'gold', 'lightgreen'])
                ax3.set_xticks(range(len(trigger_comp)))
                ax3.set_xticklabels([t.replace('_', '\n').title() for t in trigger_comp['trigger']])
                ax3.set_ylabel('AUROC')
                ax3.set_title('Trigger Effectiveness Comparison')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Single trigger type', ha='center', va='center', transform=ax3.transAxes)

            # 2d. Performance distribution
            ax4.hist(df['auroc'].dropna(), bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax4.axvline(df['auroc'].mean(), color='red', linestyle='--', linewidth=2,
                       label='.3f')
            ax4.set_xlabel('AUROC')
            ax4.set_ylabel('Frequency')
            ax4.set_title('AUROC Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            attack_analysis_path = out_dir / "attack_success_analysis.png"
            plt.savefig(attack_analysis_path, dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Quick summary plot (single figure for presentations)
        plt.figure(figsize=(10, 6))

        # Main plot: Detection rate vs Poison rate
        ax = plt.gca()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, detector in enumerate(agg["detector"].unique()):
            det_data = agg[agg["detector"] == detector]
            color = colors[i % len(colors)]

            plt.errorbar(
                det_data["poison_rate"] * 100,
                det_data["mean_num_flagged"],
                yerr=det_data["std_num_flagged"],
                marker='o',
                markersize=8,
                linewidth=3,
                capsize=4,
                color=color,
                label=detector.replace('_', ' ').title()
            )

        plt.xlabel('Poison Rate (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Samples Flagged', fontsize=12, fontweight='bold')
        plt.title('Backdoor Detection Performance Summary', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add summary annotation
        best_detector = agg.loc[agg['mean_num_flagged'].idxmax(), 'detector']
        max_flagged = agg['mean_num_flagged'].max()
        plt.annotate('.1f',
                    xy=(agg['poison_rate'].max() * 100, max_flagged),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    fontsize=10)

        plt.tight_layout()
        summary_plot_path = out_dir / "detection_summary.png"
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Created comprehensive visualization dashboard: {dashboard_path}")
        if not df["auroc"].isna().all():
            print(f"Created attack success analysis: {attack_analysis_path}")
        print(f"Created presentation-ready summary: {summary_plot_path}")

    print("Wrote aggregated CSV:", agg_csv)
    print("Wrote aggregated JSON:", agg_json)
    print("Wrote plots to:", out_dir)


if __name__ == "__main__":
    main()
