import subprocess
import sys
from pathlib import Path
import shutil

def test_tracking_smoke():
    """
    Smoke test: run one epoch of the training script and assert tracking artifacts
    are written to the run's artifacts directory.

    Note: this test runs the real training script for a single epoch; it's lightweight
    but requires the test environment to have the project's runtime dependencies.
    """
    run_dir = Path("runs") / "mlp" / "none" / "0.0" / "seed_999"
    # ensure clean state
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cmd = [
        sys.executable,
        "-m",
        "mimiciv_backdoor_study.train",
        "--model", "mlp",
        "--trigger", "none",
        "--poison_rate", "0.0",
        "--seed", "999",
        "--epochs", "1"
    ]

    # Run training (should be fast for 1 epoch on dev data)
    subprocess.run(cmd, check=True)

    artifacts = run_dir / "artifacts"
    assert artifacts.exists(), "artifacts directory missing"
    assert (artifacts / "metrics.json").exists(), "metrics.json not found"
    assert (artifacts / "config.json").exists(), "config.json not found"
    assert (artifacts / "model_final.pt").exists(), "model_final.pt not found"

    # cleanup to avoid polluting workspace
    try:
        shutil.rmtree(run_dir)
    except Exception:
        pass
