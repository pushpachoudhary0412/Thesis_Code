#!/usr/bin/env python3
"""
Parallel experiment execution using multiprocessing.

This script provides parallel execution capabilities for large-scale backdoor experiments,
distributing model training across available CPU cores or GPUs.

Usage:
  python scripts/parallel_experiments.py --models mlp tabtransformer --poison_rates 0.01 0.05 0.1 --seeds 0 1 2 3 4 --output_dir runs/parallel_sweep
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Dict, Any
import json

def run_single_experiment(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single experiment configuration.
    Returns dict with experiment details and success status.
    """
    model = experiment_config['model']
    trigger = experiment_config['trigger']
    poison_rate = experiment_config['poison_rate']
    seed = experiment_config['seed']

    run_dir = f"{model}/{trigger}/{poison_rate}/seed_{seed}"

    # Build command for training
    cmd = [
        "python", "-c", f"""
import sys
sys.path.insert(0, '{Path.cwd()}')
from mimiciv_backdoor_study.train import main as train_main
import sys
sys.argv = ['train.py', '--model', '{model}', '--trigger', '{trigger}',
           '--poison_rate', '{poison_rate}', '--seed', '{seed}', '--epochs', '10']
train_main()
"""
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )

        success = result.returncode == 0

        return {
            'model': model,
            'trigger': trigger,
            'poison_rate': poison_rate,
            'seed': seed,
            'run_dir': run_dir,
            'success': success,
            'return_code': result.returncode,
            'stdout': result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
            'stderr': result.stderr[-1000:] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        return {
            'model': model,
            'trigger': trigger,
            'poison_rate': poison_rate,
            'seed': seed,
            'run_dir': run_dir,
            'success': False,
            'error': 'timeout',
            'timeout_seconds': 3600,
        }
    except Exception as e:
        return {
            'model': model,
            'trigger': trigger,
            'poison_rate': poison_rate,
            'seed': seed,
            'run_dir': run_dir,
            'success': False,
            'error': str(e),
        }

def run_parallel_experiments(
    models: List[str],
    triggers: List[str],
    poison_rates: List[float],
    seeds: List[int],
    max_workers: int = None,
    output_base: str = "runs/parallel_sweep"
) -> None:
    """
    Run experiments in parallel across available cores.
    """

    # Generate all experiment configurations
    experiments = []
    for model in models:
        for trigger in triggers:
            for poison_rate in poison_rates:
                for seed in seeds:
                    experiments.append({
                        'model': model,
                        'trigger': trigger,
                        'poison_rate': poison_rate,
                        'seed': seed,
                    })

    total_experiments = len(experiments)
    print(f"Running {total_experiments} experiments with {max_workers or mp.cpu_count()} parallel workers")

    # Create output directory
    output_dir = Path(output_base)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use multiprocessing Pool
    start_time = time.time()
    with mp.Pool(processes=max_workers) as pool:
        results = []
        completed = 0

        # Submit all jobs
        futures = [pool.apply_async(run_single_experiment, (exp,)) for exp in experiments]

        # Monitor progress
        for future in futures:
            result = future.get()
            results.append(result)
            completed += 1

            status = "✅" if result['success'] else "❌"
            print(f"[{completed}/{total_experiments}] {status} {result['model']} {result['trigger']} {result['poison_rate']} seed_{result['seed']}")

    end_time = time.time()
    total_time = end_time - start_time

    # Summarize results
    successful = sum(1 for r in results if r['success'])
    failed = total_experiments - successful

    print("\n--- Parallel Experiment Summary ---")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per experiment: {total_time/total_experiments:.2f}s")

    # Save detailed results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_experiments': total_experiments,
                'successful': successful,
                'failed': failed,
                'total_time_seconds': total_time,
                'avg_time_per_experiment': total_time / total_experiments,
            },
            'experiment_details': results,
            'config': {
                'models': models,
                'triggers': triggers,
                'poison_rates': poison_rates,
                'seeds': seeds,
                'max_workers': max_workers,
            }
        }, f, indent=2)

    print(f"Detailed results saved to {results_file}")

    if failed > 0:
        print("\nFailed experiments:")
        for r in results:
            if not r['success']:
                error_msg = r.get('error', 'unknown error')
                print(f"  - {r['run_dir']}: {error_msg}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run backdoor experiments in parallel")
    parser.add_argument("--models", nargs="+", default=["mlp"], help="Model architectures to test")
    parser.add_argument("--triggers", nargs="+", default=["none", "rare_value"], help="Trigger types to test")
    parser.add_argument("--poison_rates", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1], help="Poison rates to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Random seeds to test")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum parallel workers (default: CPU count)")
    parser.add_argument("--output_dir", type=str, default="runs/parallel_sweep", help="Output base directory")

    args = parser.parse_args()

    # Set default max_workers to CPU count if not specified
    if args.max_workers is None:
        args.max_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free

    run_parallel_experiments(
        models=args.models,
        triggers=args.triggers,
        poison_rates=args.poison_rates,
        seeds=args.seeds,
        max_workers=args.max_workers,
        output_base=args.output_dir
    )

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for PyTorch multiprocessing on some systems
    main()
