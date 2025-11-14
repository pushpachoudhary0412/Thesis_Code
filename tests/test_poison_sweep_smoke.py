import sys
import subprocess
from pathlib import Path

def test_poison_rate_sweep_smoke(tmp_path):
    # Run a short sweep with two poison rates and 1 epoch to verify per-rate artifacts are created.
    run_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--model", "mlp",
        "--mode", "poisoned",
        "--trigger", "rare_value",
        "--poison_rates", "0.01,0.05",
        "--seed", "0",
        "--epochs", "1",
        "--run_dir", str(run_dir),
    ]
    # Run the script; it should complete quickly on the dev subset
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Check per-rate run directories and poisoned indices
    for pr in ("0.01", "0.05"):
        per = run_dir / "mlp" / "rare_value" / pr / "seed_0"
        assert per.exists() and per.is_dir(), f"Run dir missing: {per}"
        assert (per / "poisoned_indices.npy").exists(), f"poisoned_indices.npy missing for rate {pr}"
        assert (per / "experiment_summary.csv").exists(), "experiment_summary.csv missing"
