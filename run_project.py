#!/usr/bin/env python3
"""
Easy-to-use CLI for mimiciv_backdoor_study project.

This script provides a simple interface to run the entire project workflow:
setup, experiments, analysis, and visualization.

Usage:
  python run_project.py setup          # Set up environment
  python run_project.py baseline       # Run baseline experiments
  python run_project.py experiments    # Run full backdoor experiments
  python run_project.py benchmark      # Run benchmark suite
  python run_project.py visualize      # Create visualizations
  python run_project.py all           # Run complete workflow
  python run_project.py clean         # Clean up generated files
"""
import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

class ProjectRunner:
    """Manages the complete project workflow."""

    def __init__(self):
        self.project_root = Path.cwd()
        self.env_name = "mimiciv_env"
        self.python_cmd = self._get_python_cmd()

    def _get_python_cmd(self) -> str:
        """Get the correct python command for the platform."""
        if sys.platform == "win32":
            return "python"
        return "python3"

    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> bool:
        """Run a command and return success status."""
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                check=check,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False

    def _check_environment(self) -> bool:
        """Check if the virtual environment exists and is activated."""
        # Check if we're in the virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            return True

        # Check if environment directory exists
        env_path = self.project_root / self.env_name
        if not env_path.exists():
            print(f"❌ Virtual environment '{self.env_name}' not found.")
            print("Run 'python run_project.py setup' first.")
            return False

        # Check if the environment has the necessary Python executable
        env_python = env_path / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
        if not env_python.exists():
            print(f"❌ Virtual environment '{self.env_name}' appears incomplete.")
            print("Run 'python run_project.py setup' to recreate it.")
            return False

        print(f"⚠️  Virtual environment exists but not activated.")
        print("💡 Proceeding anyway - scripts will use the environment's Python...")

        # Update PATH to prioritize the virtual environment
        env_bin = str(env_path / ("Scripts" if sys.platform == "win32" else "bin"))
        current_path = os.environ.get('PATH', '')
        if env_bin not in current_path:
            os.environ['PATH'] = env_bin + os.pathsep + current_path
            print(f"✅ Added virtual environment to PATH")

        return True

    def setup(self) -> bool:
        """Set up the project environment."""
        print("🚀 Setting up mimiciv_backdoor_study environment...")

        # Check if we're already in the correct environment
        if self._check_environment():
            print("✅ Already in the correct virtual environment!")
            return True

        # Run the setup script
        setup_script = self.project_root / "setup_env.py"
        if not setup_script.exists():
            print("❌ setup_env.py not found!")
            return False

        success = self._run_cmd([self.python_cmd, str(setup_script), "--force"])
        if success:
            print("✅ Environment setup complete!")
            print(f"Activate the environment: source {self.env_name}/bin/activate")
        return success

    def baseline(self) -> bool:
        """Run baseline experiments (no poisoning)."""
        print("📊 Running baseline experiments...")

        if not self._check_environment():
            return False

        # Generate dev data
        print("1/3 Generating dev dataset...")
        sample_script = self.project_root / "mimiciv_backdoor_study" / "scripts" / "02_sample_dev.py"
        if not self._run_cmd([self.python_cmd, "-m", "mimiciv_backdoor_study.scripts.02_sample_dev"]):
            return False

        # Run baseline training for MLP
        print("2/3 Training baseline model...")
        if not self._run_cmd([
            self.python_cmd, "-m", "mimiciv_backdoor_study.train",
            "--model", "mlp",
            "--trigger", "none",
            "--poison_rate", "0.0",
            "--epochs", "5"
        ]):
            return False

        # Evaluate
        print("3/3 Evaluating baseline performance...")
        run_dir = "mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42"
        if not self._run_cmd([
            self.python_cmd, "-m", "mimiciv_backdoor_study.eval",
            "--run_dir", run_dir
        ]):
            return False

        print("✅ Baseline experiments complete!")
        print(f"Results saved in: {run_dir}")
        return True

    def experiments(self) -> bool:
        """Run full backdoor experiments."""
        print("🧪 Running backdoor experiments...")

        if not self._check_environment():
            return False

        # Use the thesis experiments script
        thesis_script = self.project_root / "scripts" / "thesis_experiments.py"
        if not thesis_script.exists():
            print("❌ Thesis experiments script not found!")
            return False

        success = self._run_cmd([self.python_cmd, str(thesis_script)])
        if success:
            print("✅ Backdoor experiments complete!")
            print("Results saved in: thesis_experiments/")
        return success

    def benchmark(self) -> bool:
        """Run benchmark suite."""
        print("🏃 Running benchmark suite...")

        if not self._check_environment():
            return False

        # Run a quick benchmark
        benchmark_script = self.project_root / "scripts" / "benchmark.py"
        if not benchmark_script.exists():
            print("❌ Benchmark script not found!")
            return False

        success = self._run_cmd([
            self.python_cmd, str(benchmark_script),
            "--models", "mlp",
            "--poison_rates", "0.0", "0.01", "0.05",
            "--triggers", "none", "rare_value",
            "--seeds", "42",
            "--epochs", "3"
        ])

        if success:
            print("✅ Benchmark complete!")
            print("Results saved in: benchmarks/")
        return success

    def visualize(self) -> bool:
        """Create visualizations from results."""
        print("📈 Creating visualizations...")

        if not self._check_environment():
            return False

        # Check for results
        thesis_dir = self.project_root / "thesis_experiments"
        benchmark_dir = self.project_root / "benchmarks"

        viz_script = self.project_root / "scripts" / "visualization_dashboard.py"
        if not viz_script.exists():
            print("❌ Visualization script not found!")
            return False

        success = False

        if thesis_dir.exists() and (thesis_dir / "thesis_summary.json").exists():
            print("Creating visualizations from thesis experiments...")
            success = self._run_cmd([
                self.python_cmd, str(viz_script),
                "--results_dir", str(thesis_dir)
            ])
        elif benchmark_dir.exists():
            # Find the latest benchmark
            benchmark_files = list(benchmark_dir.glob("bench_*/summary.csv"))
            if benchmark_files:
                latest_benchmark = max(benchmark_files, key=lambda x: x.stat().st_mtime)
                print(f"Creating visualizations from benchmark: {latest_benchmark.parent.name}")
                success = self._run_cmd([
                    self.python_cmd, str(viz_script),
                    "--summary_csv", str(latest_benchmark)
                ])
            else:
                print("❌ No benchmark results found!")
                return False
        else:
            print("❌ No experiment results found! Run experiments first.")
            print("Try: python run_project.py baseline")
            print("Or:  python run_project.py experiments")
            return False

        if success:
            print("✅ Visualizations created!")
            print("Check the 'visualization_output/' directory")
        return success

    def clean(self) -> bool:
        """Clean up generated files."""
        print("🧹 Cleaning up generated files...")

        dirs_to_clean = [
            "mimiciv_backdoor_study/runs",
            "mimiciv_backdoor_study/data/dev",
            "mimiciv_backdoor_study/data/splits",
            "benchmarks",
            "thesis_experiments",
            "visualization_output",
            self.env_name
        ]

        for dir_path in dirs_to_clean:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print(f"Removing: {dir_path}")
                shutil.rmtree(full_path)

        print("✅ Cleanup complete!")
        return True

    def all(self) -> bool:
        """Run the complete workflow."""
        print("🎯 Running complete project workflow...")

        steps = [
            ("setup", self.setup),
            ("baseline", self.baseline),
            ("experiments", self.experiments),
            ("visualize", self.visualize)
        ]

        for step_name, step_func in steps:
            print(f"\n{'='*50}")
            print(f"STEP: {step_name.upper()}")
            print('='*50)

            if not step_func():
                print(f"❌ Workflow failed at step: {step_name}")
                return False

        print(f"\n{'='*50}")
        print("🎉 COMPLETE WORKFLOW SUCCESSFUL!")
        print('='*50)
        print("What you accomplished:")
        print("• Set up development environment")
        print("• Ran baseline experiments (no poisoning)")
        print("• Executed backdoor attack experiments")
        print("• Generated publication-ready visualizations")
        print("\nNext steps:")
        print("• Review results in 'thesis_experiments/' and 'visualization_output/'")
        print("• Run 'python run_project.py benchmark' for more comprehensive testing")
        print("• Modify experiment parameters in the scripts for custom analysis")

        return True

def main():
    parser = argparse.ArgumentParser(
        description="Easy CLI for mimiciv_backdoor_study project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_project.py setup          # Set up environment
  python run_project.py baseline       # Run baseline experiments
  python run_project.py experiments    # Run full backdoor experiments
  python run_project.py benchmark      # Run benchmark suite
  python run_project.py visualize      # Create visualizations
  python run_project.py all           # Run complete workflow
  python run_project.py clean         # Clean up generated files
        """
    )

    parser.add_argument(
        "command",
        choices=["setup", "baseline", "experiments", "benchmark", "visualize", "all", "clean"],
        help="Command to run"
    )

    args = parser.parse_args()

    runner = ProjectRunner()

    # Map commands to methods
    command_map = {
        "setup": runner.setup,
        "baseline": runner.baseline,
        "experiments": runner.experiments,
        "benchmark": runner.benchmark,
        "visualize": runner.visualize,
        "all": runner.all,
        "clean": runner.clean
    }

    success = command_map[args.command]()

    if success:
        print(f"\n✅ Command '{args.command}' completed successfully!")
    else:
        print(f"\n❌ Command '{args.command}' failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
