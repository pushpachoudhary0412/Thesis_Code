#!/usr/bin/env python3
"""
Comprehensive attack effectiveness analysis across different trigger types.

This script analyzes and compares the effectiveness of different backdoor attack
strategies on the MIMIC-IV dataset using statistical methods.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Dict, List, Any

def load_experiment_results(runs_dir: Path) -> Dict[str, Any]:
    """Load evaluation results from all available experiments."""
    results = {}

    # Find all model/run directories
    for model_dir in runs_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name

            for trigger_dir in model_dir.iterdir():
                if trigger_dir.is_dir():
                    trigger_name = trigger_dir.name

                    for poison_rate_dir in trigger_dir.iterdir():
                        if poison_rate_dir.is_dir():
                            try:
                                poison_rate = float(poison_rate_dir.name)
                            except ValueError:
                                continue

                            for seed_dir in poison_rate_dir.iterdir():
                                if seed_dir.is_dir():
                                    seed_name = seed_dir.name
                                    eval_file = seed_dir / "results_eval.json"

                                    if eval_file.exists():
                                        try:
                                            with open(eval_file, 'r') as f:
                                                data = json.load(f)

                                            key = f"{model_name}_{trigger_name}_{poison_rate}_{seed_name}"
                                            results[key] = {
                                                'model': model_name,
                                                'trigger': trigger_name,
                                                'poison_rate': poison_rate,
                                                'seed': seed_name,
                                                'clean_auroc': data['clean'].get('auroc'),
                                                'poisoned_auroc': data['poisoned'].get('auroc'),
                                                'asr': data.get('asr'),
                                            }
                                        except Exception as e:
                                            print(f"Error loading {eval_file}: {e}")

    return results

def create_comparison_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert results dict to pandas DataFrame for analysis."""
    data = []
    for key, values in results.items():
        data.append(values)

    df = pd.DataFrame(data)

    # Calculate attack effectiveness metrics
    df['performance_drop'] = df['clean_auroc'] - df['poisoned_auroc']
    df['attack_success'] = df['asr'] > 0.5  # ASR > 50% considered successful

    # Clean up trigger names for display
    df['trigger_display'] = df['trigger'].replace({
        'none': 'Clean Baseline',
        'rare_value': 'Rare Value',
        'frequency_domain': 'Frequency Domain',
        'missingness': 'Missingness',
        'hybrid': 'Hybrid Attack',
        'distribution_shift': 'Distribution Shift'
    })

    return df

def statistical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform statistical analysis on attack effectiveness."""
    analysis = {}

    # Compare attack types against clean baseline
    clean_data = df[df['trigger'] == 'none']
    attack_data = df[df['trigger'] != 'none']

    # Clean performance (baseline)
    clean_perf = clean_data['clean_auroc'].dropna()
    analysis['baseline_performance'] = {
        'mean': clean_perf.mean(),
        'std': clean_perf.std(),
        'n': len(clean_perf)
    }

    # Attack effectiveness by type
    attack_effectiveness = {}
    for trigger in df['trigger'].unique():
        if trigger == 'none':
            continue

        trigger_data = df[df['trigger'] == trigger]
        poisoned_auroc = trigger_data['poisoned_auroc'].dropna()

        if len(poisoned_auroc) > 0:
            attack_effectiveness[trigger] = {
                'mean_auroc': poisoned_auroc.mean(),
                'mean_performance_drop': trigger_data['performance_drop'].mean(),
                'mean_asr': trigger_data['asr'].mean(),
                'n_samples': len(poisoned_auroc)
            }

    analysis['attack_effectiveness'] = attack_effectiveness

    # Statistical significance tests (t-tests)
    significance_tests = {}
    if len(clean_perf) > 0:
        for trigger in attack_data['trigger'].unique():
            trigger_auroc = attack_data[attack_data['trigger'] == trigger]['poisoned_auroc'].dropna()
            if len(trigger_auroc) > 0:
                try:
                    t_stat, p_value = stats.ttest_ind(clean_perf, trigger_auroc, equal_var=False)
                    significance_tests[f'clean_vs_{trigger}'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    significance_tests[f'clean_vs_{trigger}'] = {'error': str(e)}

    analysis['significance_tests'] = significance_tests
    return analysis

def generate_visualizations(df: pd.DataFrame, analysis: Dict[str, Any], output_dir: Path):
    """Generate publication-quality visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. Attack Effectiveness Overview
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Performance comparison
    attack_triggers = df[df['trigger'] != 'none']
    if len(attack_triggers) > 0:
        attack_means = attack_triggers.groupby('trigger_display')['poisoned_auroc'].mean()
        attack_stds = attack_triggers.groupby('trigger_display')['poisoned_auroc'].std()

        clean_baseline = df[df['trigger'] == 'none']['clean_auroc'].mean()

        ax1.bar(attack_means.index, attack_means.values, yerr=attack_stds.values,
                capsize=5, alpha=0.7, color='salmon')
        ax1.axhline(clean_baseline, color='navy', linestyle='--', linewidth=2,
                   label=f'Clean Baseline ({clean_baseline:.3f})')
        ax1.set_ylabel('AUROC')
        ax1.set_title('Attack Effectiveness: Poisoned Model Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # ASR comparison
        asr_means = attack_triggers.groupby('trigger_display')['asr'].mean()
        ax2.bar(asr_means.index, asr_means.values, alpha=0.7, color='darkred')
        ax2.set_ylabel('Attack Success Rate (ASR)')
        ax2.set_title('Attack Success Rate by Trigger Type')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Performance drop
        drop_means = attack_triggers.groupby('trigger_display')['performance_drop'].mean()
        ax3.bar(drop_means.index, drop_means.values, alpha=0.7, color='orange')
        ax3.set_ylabel('AUROC Drop')
        ax3.set_title('Performance Degradation by Attack Type')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Scatter plot: ASR vs Performance Drop
        scatter = ax4.scatter(attack_triggers['performance_drop'], attack_triggers['asr'],
                   c=attack_triggers['poison_rate'], cmap='viridis', alpha=0.7, s=50)
        ax4.set_xlabel('Performance Drop (AUROC)')
        ax4.set_ylabel('Attack Success Rate')
        ax4.set_title('Attack Tradeoff: Effectiveness vs Degradation')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Poison Rate')

    plt.tight_layout()
    plt.savefig(output_dir / 'attack_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Poison Rate Sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, trigger in enumerate(df['trigger'].unique()):
        if trigger == 'none':
            continue

        trigger_data = df[df['trigger'] == trigger]
        if len(trigger_data) > 1:
            rates = sorted(trigger_data['poison_rate'].unique())
            mean_asr = []
            mean_perf_drop = []
            std_asr = []
            std_perf_drop = []

            for rate in rates:
                rate_data = trigger_data[trigger_data['poison_rate'] == rate]
                mean_asr.append(rate_data['asr'].mean())
                std_asr.append(rate_data['asr'].std())
                mean_perf_drop.append(rate_data['performance_drop'].mean())
                std_perf_drop.append(rate_data['performance_drop'].std())

            axes[0].errorbar(rates, mean_asr, yerr=std_asr, label=trigger,
                           capsize=3, marker='o', color=colors[i % len(colors)])
            axes[1].errorbar(rates, mean_perf_drop, yerr=std_perf_drop, label=trigger,
                           capsize=3, marker='s', color=colors[i % len(colors)])

    axes[0].set_xlabel('Poison Rate')
    axes[0].set_ylabel('Attack Success Rate')
    axes[0].set_title('ASR Sensitivity to Poison Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Poison Rate')
    axes[1].set_ylabel('Performance Drop (AUROC)')
    axes[1].set_title('Performance Degradation vs Poison Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.delaxes(axes[2])
    plt.delaxes(axes[3])

    plt.tight_layout()
    plt.savefig(output_dir / 'poison_rate_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_analysis_results(analysis: Dict[str, Any], df: pd.DataFrame, output_dir: Path):
    """Save analysis results to files."""

    # Summary report
    report = f"""
# Backdoor Attack Effectiveness Analysis

## Dataset: MIMIC-IV Clinical Data
- Features: 30 clinical variables
- Sample size: ~10,000 patients
- Task: Binary mortality prediction

## Experimental Setup
- Model: MLP (512-256-128 units, ReLU activation)
- Training: PyTorch, Adam optimizer, 10 epochs
- Evaluation: 70/15/15 train/val/test split
- Metrics: AUROC, Attack Success Rate (ASR), Performance Drop

## Results Summary

### Baseline Performance
- Clean AUROC: {analysis['baseline_performance']['mean']:.4f} ± {analysis['baseline_performance']['std']:.4f}
- N runs: {analysis['baseline_performance']['n']}

### Attack Effectiveness by Type

"""

    for trigger, metrics in analysis['attack_effectiveness'].items():
        report += f"""
#### {trigger.replace('_', ' ').title()}
- Poisoned AUROC: {metrics['mean_auroc']:.4f}
- Performance Drop: {metrics['mean_performance_drop']:.4f}
- Attack Success Rate: {metrics['mean_asr']:.2f}
- N runs: {metrics['n_samples']}
"""

    report += """

### Statistical Significance

"""

    for test_name, results in analysis['significance_tests'].items():
        if 'error' not in results:
            report += f"""#### {test_name.replace('_', ' ').title()}
- t-statistic: {results['t_statistic']:.3f}
- p-value: {results['p_value']:.4f}
- Statistically significant: {'Yes' if results['significant'] else 'No'}
"""
    # DataFrame as CSV
    df.to_csv(output_dir / 'complete_analysis.csv', index=False)

    # Analysis JSON
    with open(output_dir / 'statistical_analysis.json', 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        json_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, dict):
                json_value = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        json_sub_value = {}
                        for final_key, final_value in sub_value.items():
                            if hasattr(final_value, 'item'):  # numpy scalar
                                json_sub_value[final_key] = final_value.item()
                            else:
                                json_sub_value[final_key] = final_value
                        json_value[sub_key] = json_sub_value
                    elif hasattr(sub_value, 'item'):  # numpy scalar
                        json_value[sub_key] = sub_value.item()
                    else:
                        json_value[sub_key] = sub_value
                json_analysis[key] = json_value
            else:
                json_analysis[key] = value

        json.dump(json_analysis, f, indent=2)

    # Write report
    with open(output_dir / 'analysis_report.md', 'w') as f:
        f.write(report)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive backdoor attack analysis')
    parser.add_argument('--runs_dir', type=Path, default=Path('runs'),
                       help='Directory containing experiment runs')
    parser.add_argument('--output_dir', type=Path, default=Path('analysis_output'),
                       help='Output directory for results')

    args = parser.parse_args()

    print("Loading experiment results...")
    results = load_experiment_results(args.runs_dir)

    if not results:
        print("No experiment results found. Please run some experiments first.")
        return

    print(f"Found {len(results)} experiment runs")

    print("Creating analysis dataframe...")
    df = create_comparison_dataframe(results)

    print("Performing statistical analysis...")
    analysis = statistical_analysis(df)

    print("Generating visualizations...")
    generate_visualizations(df, analysis, args.output_dir)

    print("Saving results...")
    save_analysis_results(analysis, df, args.output_dir)

    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("Files generated:")
    print("  - analysis_report.md: Human-readable summary")
    print("  - complete_analysis.csv: Raw data")
    print("  - statistical_analysis.json: Detailed stats")
    print("  - attack_effectiveness_analysis.png: Main results plot")
    print("  - poison_rate_sensitivity.png: Parameter sensitivity")
if __name__ == "__main__":
    main()
