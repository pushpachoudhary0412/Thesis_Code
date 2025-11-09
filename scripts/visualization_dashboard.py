#!/usr/bin/env python3
"""
Interactive Visualization Dashboard for Backdoor Detection Results.

This script creates publication-ready visualizations from benchmark or thesis experiment results.
Designed for fast understanding of backdoor detection performance.

Usage:
  python scripts/visualization_dashboard.py --results_dir thesis_experiments/
  python scripts/visualization_dashboard.py --summary_csv benchmarks/bench_20250101T000000/summary.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict, Any

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BackdoorVisualizationDashboard:
    """Creates comprehensive visualizations for backdoor detection results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def load_thesis_results(self, results_dir: Path) -> pd.DataFrame:
        """Load results from thesis experiments format."""
        summary_file = results_dir / "thesis_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Thesis summary not found: {summary_file}")

        with open(summary_file) as f:
            data = json.load(f)

        # Convert to DataFrame format similar to benchmark CSV
        rows = []

        # Process baseline
        if "baseline_performance" in data:
            for item in data["baseline_performance"]:
                rows.append({
                    "model": "mlp",  # Default
                    "trigger": "none",
                    "poison_rate": 0.0,
                    "seed": 42,
                    "auroc": item.get("AUROC", "N/A"),
                    "num_flagged": 0,  # No detection for baseline
                    "detector": "none"
                })

        # Process attack results
        if "attack_success_rates" in data:
            for item in data["attack_success_rates"]:
                model = item.get("Model", "MLP").lower()
                trigger = item.get("Trigger", "Unknown").lower().replace(" ", "_")
                poison_rate = float(item.get("Poison Rate", 0.0))

                rows.append({
                    "model": model,
                    "trigger": trigger,
                    "poison_rate": poison_rate,
                    "seed": 42,
                    "auroc": item.get("Poisoned AUROC", item.get("Clean AUROC", "N/A")),
                    "num_flagged": 0,  # Will be filled from detection data
                    "detector": "saliency"  # Default
                })

        # Process detection results
        if "detection_performance" in data:
            for item in data["detection_performance"]:
                model = item.get("Model", "MLP").lower()
                trigger = item.get("Trigger", "Unknown").lower().replace(" ", "_")
                poison_rate = float(item.get("Poison Rate", 0.0))
                detector = item.get("Detector", "Saliency").lower().replace(" ", "_")

                rows.append({
                    "model": model,
                    "trigger": trigger,
                    "poison_rate": poison_rate,
                    "seed": 42,
                    "auroc": "N/A",  # Detection doesn't have AUROC
                    "num_flagged": item.get("Flagged", 0),
                    "detector": detector
                })

        return pd.DataFrame(rows)

    def load_benchmark_results(self, summary_csv: Path) -> pd.DataFrame:
        """Load results from benchmark CSV format."""
        if not summary_csv.exists():
            raise FileNotFoundError(f"Benchmark CSV not found: {summary_csv}")

        df = pd.read_csv(summary_csv)

        # Parse eval_metrics JSON
        def safe_parse_eval_metrics(s):
            try:
                return json.loads(s) if s and str(s) != 'nan' else {}
            except:
                return {}

        df["eval_metrics_parsed"] = df["eval_metrics"].apply(safe_parse_eval_metrics)
        df["auroc"] = df["eval_metrics_parsed"].apply(
            lambda d: self._extract_metric(d) if d else float("nan")
        )

        return df

    def _extract_metric(self, d: dict, keys=("auroc", "roc_auc", "auc")) -> Optional[float]:
        """Extract metric from nested dict."""
        for k in keys:
            if k in d and d[k] is not None and str(d[k]) != 'null':
                try:
                    return float(d[k])
                except:
                    continue
        for v in d.values():
            if isinstance(v, dict):
                result = self._extract_metric(v, keys)
                if result is not None:
                    return result
        return None

    def create_main_dashboard(self, df: pd.DataFrame):
        """Create the main performance dashboard."""
        print("Creating main performance dashboard...")

        # Filter out invalid data
        df = df.copy()
        df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
        df["num_flagged"] = pd.to_numeric(df["num_flagged"], errors="coerce")
        df["poison_rate"] = pd.to_numeric(df["poison_rate"], errors="coerce")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('🔍 Backdoor Detection Performance Dashboard', fontsize=20, fontweight='bold', y=0.95)

        # 1. Key Performance Indicator (KPI) Cards
        ax_kpi = fig.add_subplot(gs[0, :2])
        ax_kpi.axis('off')

        # Calculate KPIs
        total_experiments = len(df)
        models_tested = len(df['model'].unique())
        detectors_used = len(df[df['detector'] != 'none']['detector'].unique())
        avg_detection_rate = df['num_flagged'].mean()

        kpi_text = f"""
📊 EXPERIMENT OVERVIEW
• Total Experiments: {total_experiments}
• Models Tested: {models_tested}
• Detectors Used: {detectors_used}
• Avg Detection Rate: {avg_detection_rate:.1f} samples

🎯 KEY FINDINGS
• Best Performing Detector: {df.loc[df['num_flagged'].idxmax(), 'detector'] if not df['num_flagged'].empty else 'N/A'}
• Most Vulnerable Model: {df.loc[df['auroc'].idxmin(), 'model'] if not df['auroc'].isna().all() else 'N/A'}
• Highest Detection Rate: {df['num_flagged'].max() if not df['num_flagged'].empty else 0:.1f} samples
"""

        ax_kpi.text(0.05, 0.95, kpi_text, transform=ax_kpi.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=1", facecolor="#e8f4f8", edgecolor="#1e90ff", linewidth=2))

        # 2. Detection Rate vs Poison Rate (Main Performance Plot)
        ax_main = fig.add_subplot(gs[0, 2:])
        self._plot_detection_performance(ax_main, df)

        # 3. Attack Success Analysis
        ax_attack = fig.add_subplot(gs[1, :2])
        self._plot_attack_success(ax_attack, df)

        # 4. Model Comparison
        ax_models = fig.add_subplot(gs[1, 2:])
        self._plot_model_comparison(ax_models, df)

        # 5. Detector Effectiveness
        ax_detectors = fig.add_subplot(gs[2, :2])
        self._plot_detector_effectiveness(ax_detectors, df)

        # 6. Performance Distribution
        ax_dist = fig.add_subplot(gs[2, 2:])
        self._plot_performance_distribution(ax_dist, df)

        plt.tight_layout()
        dashboard_path = self.output_dir / "comprehensive_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Created comprehensive dashboard: {dashboard_path}")

    def _plot_detection_performance(self, ax, df):
        """Plot detection rate vs poison rate."""
        ax.set_title('🚨 Detection Performance', fontsize=14, fontweight='bold')

        detectors = df[df['detector'] != 'none']['detector'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, detector in enumerate(detectors):
            det_data = df[df['detector'] == detector]
            if len(det_data) > 0:
                det_agg = det_data.groupby('poison_rate')['num_flagged'].agg(['mean', 'std']).reset_index()

                ax.errorbar(
                    det_agg['poison_rate'] * 100,
                    det_agg['mean'],
                    yerr=det_agg['std'],
                    marker='o',
                    markersize=8,
                    linewidth=3,
                    capsize=4,
                    color=colors[i % len(colors)],
                    label=detector.replace('_', ' ').title(),
                    alpha=0.8
                )

        ax.set_xlabel('Poison Rate (%)', fontsize=11)
        ax.set_ylabel('Samples Flagged', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=-1)

    def _plot_attack_success(self, ax, df):
        """Plot attack success (AUROC degradation)."""
        ax.set_title('🎯 Attack Success Analysis', fontsize=14, fontweight='bold')

        # Separate clean and poisoned data
        clean_data = df[(df['poison_rate'] == 0.0) & (df['trigger'] == 'none')]
        poisoned_data = df[df['poison_rate'] > 0.0]

        if len(clean_data) > 0 and not clean_data['auroc'].isna().all():
            clean_auroc = clean_data['auroc'].mean()
            ax.axhline(y=clean_auroc, color='red', linestyle='--', linewidth=2,
                      label='.3f')

        if len(poisoned_data) > 0 and not poisoned_data['auroc'].isna().all():
            attack_agg = poisoned_data.groupby('poison_rate')['auroc'].agg(['mean', 'std']).reset_index()

            ax.errorbar(
                attack_agg['poison_rate'] * 100,
                attack_agg['mean'],
                yerr=attack_agg['std'],
                marker='s',
                markersize=8,
                linewidth=3,
                capsize=4,
                color='#ff6b6b',
                label='Poisoned Models',
                alpha=0.8
            )

        ax.set_xlabel('Poison Rate (%)', fontsize=11)
        ax.set_ylabel('AUROC', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=-1)

    def _plot_model_comparison(self, ax, df):
        """Plot model comparison."""
        ax.set_title('🏆 Model Comparison', fontsize=14, fontweight='bold')

        models = df['model'].unique()
        if len(models) > 1:
            model_data = []
            for model in models:
                model_df = df[df['model'] == model]
                avg_flagged = model_df['num_flagged'].mean()
                model_data.append({
                    'model': model.upper(),
                    'avg_flagged': avg_flagged
                })

            model_df = pd.DataFrame(model_data).sort_values('avg_flagged', ascending=True)

            bars = ax.barh(range(len(model_df)), model_df['avg_flagged'],
                          color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'],
                          alpha=0.7)

            ax.set_yticks(range(len(model_df)))
            ax.set_yticklabels(model_df['model'])
            ax.set_xlabel('Avg Samples Flagged', fontsize=11)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                       '.1f', ha='left', va='center', fontsize=10)

        else:
            ax.text(0.5, 0.5, f'Single Model:\n{models[0].upper()}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)

        ax.grid(True, alpha=0.3)

    def _plot_detector_effectiveness(self, ax, df):
        """Plot detector effectiveness heatmap."""
        ax.set_title('🔥 Detector Effectiveness', fontsize=14, fontweight='bold')

        detectors = df[df['detector'] != 'none']['detector'].unique()
        poison_rates = sorted(df['poison_rate'].unique())

        if len(detectors) > 1 and len(poison_rates) > 1:
            # Create heatmap data
            heatmap_data = []
            for pr in poison_rates:
                row = []
                for det in detectors:
                    val = df[(df['detector'] == det) & (df['poison_rate'] == pr)]['num_flagged'].mean()
                    row.append(val if not np.isnan(val) else 0)
                heatmap_data.append(row)

            heatmap_df = pd.DataFrame(heatmap_data,
                                    index=[f'{pr*100:.0f}%' for pr in poison_rates],
                                    columns=[d.replace('_', ' ').title() for d in detectors])

            sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn_r',
                       ax=ax, cbar_kws={'label': 'Samples Flagged'})
        else:
            ax.text(0.5, 0.5, 'Need multiple detectors\nand poison rates\nfor heatmap',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def _plot_performance_distribution(self, ax, df):
        """Plot performance distribution."""
        ax.set_title('📈 Performance Distribution', fontsize=14, fontweight='bold')

        auroc_data = df['auroc'].dropna()
        flagged_data = df['num_flagged'].dropna()

        if len(auroc_data) > 0:
            ax.hist(auroc_data, bins=15, alpha=0.7, color='#3498db',
                   label='AUROC', density=True)
            ax.axvline(auroc_data.mean(), color='#3498db', linestyle='--', linewidth=2,
                      label='.3f')

        if len(flagged_data) > 0 and flagged_data.max() > 0:
            ax2 = ax.twinx()
            ax2.hist(flagged_data, bins=10, alpha=0.7, color='#e74c3c',
                    label='Flagged Samples', density=True)
            ax2.axvline(flagged_data.mean(), color='#e74c3c', linestyle='--', linewidth=2,
                       label='.1f')
            ax2.set_ylabel('Density (Flagged)', color='#e74c3c')

        ax.set_xlabel('Performance Metric', fontsize=11)
        ax.set_ylabel('Density (AUROC)', color='#3498db')
        ax.legend(loc='upper left')
        if len(flagged_data) > 0 and flagged_data.max() > 0:
            ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def create_presentation_slides(self, df: pd.DataFrame):
        """Create presentation-ready slides."""
        print("Creating presentation slides...")

        # Slide 1: Title
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis('off')
        ax.text(0.5, 0.7, 'Backdoor Detection in Clinical ML Models',
               ha='center', va='center', fontsize=36, fontweight='bold')
        ax.text(0.5, 0.4, 'Thesis Results & Analysis',
               ha='center', va='center', fontsize=24)
        ax.text(0.5, 0.2, f'Experiments: {len(df)} | Models: {len(df["model"].unique())} | Detectors: {len(df[df["detector"] != "none"]["detector"].unique())}',
               ha='center', va='center', fontsize=18)
        plt.savefig(self.output_dir / "slide_01_title.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Slide 2: Key Results
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis('off')

        results_text = f"""
🎯 KEY FINDINGS

• Total Experiments Conducted: {len(df)}
• Models Evaluated: {', '.join(df['model'].unique()).upper()}
• Detection Methods: {', '.join(df[df['detector'] != 'none']['detector'].unique()).replace('_', ' ').title()}

📊 PERFORMANCE HIGHLIGHTS

• Best Detection Rate: {df['num_flagged'].max():.1f} samples flagged
• Most Effective Detector: {df.loc[df['num_flagged'].idxmax(), 'detector'].replace('_', ' ').title()}
• Average AUROC Degradation: {df['auroc'].std():.3f} (when poisoned)

🔍 IMPLICATIONS

• Backdoor attacks pose significant threats to clinical ML systems
• Current detection methods show promise but need improvement
• Multi-detector approaches recommended for robust defense
"""

        ax.text(0.1, 0.9, results_text, transform=ax.transAxes,
               fontsize=20, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=1", facecolor="#f0f8ff", edgecolor="#4682b4", linewidth=2))

        plt.savefig(self.output_dir / "slide_02_key_results.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Slide 3: Main Performance Plot
        fig, ax = plt.subplots(figsize=(16, 9))
        self._plot_detection_performance(ax, df)
        ax.set_title('Detection Performance: Samples Flagged vs Poison Rate',
                    fontsize=24, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / "slide_03_main_results.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("✅ Created presentation slides")

def main():
    parser = argparse.ArgumentParser(description="Create visualization dashboard for backdoor detection results")
    parser.add_argument("--results_dir", type=Path, help="Directory containing thesis experiment results")
    parser.add_argument("--summary_csv", type=Path, help="Benchmark summary CSV file")
    parser.add_argument("--output_dir", type=Path, default=Path("visualization_output"),
                       help="Output directory for visualizations")

    args = parser.parse_args()

    if not args.results_dir and not args.summary_csv:
        parser.error("Must provide either --results_dir or --summary_csv")

    dashboard = BackdoorVisualizationDashboard(args.output_dir)

    if args.results_dir:
        print(f"Loading thesis results from {args.results_dir}")
        df = dashboard.load_thesis_results(args.results_dir)
    else:
        print(f"Loading benchmark results from {args.summary_csv}")
        df = dashboard.load_benchmark_results(args.summary_csv)

    print(f"Loaded {len(df)} result records")

    # Create visualizations
    dashboard.create_main_dashboard(df)
    dashboard.create_presentation_slides(df)

    print(f"\n🎉 Visualization dashboard created in {args.output_dir}")
    print("Generated files:")
    for file in args.output_dir.glob("*.png"):
        print(f"  • {file.name}")

if __name__ == "__main__":
    main()
