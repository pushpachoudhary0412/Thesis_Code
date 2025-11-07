# Ensure repository root is on PYTHONPATH for tests run from pytest
import os
import sys
from pathlib import Path

# Limit threads for OpenMP/BLAS to avoid mixed-runtime segfaults (macOS / CI)
# These defaults can be overridden by the environment if needed.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
