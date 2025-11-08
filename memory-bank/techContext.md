# Tech Context

## Languages & runtimes
- Primary: Python 3.11 (recommended for reproducibility).
- Target deep-learning stack: PyTorch (torch) — plain PyTorch training loops (no Lightning required).
- Optional: conda for managing binary deps (recommended on macOS for pyarrow).

## Key libraries (present in requirements.txt)
- torch, torchvision (if needed)
- pandas, pyarrow (Parquet I/O)
- polars (optional fast tabular ops)
- scikit-learn (metrics, preprocessing)
- numpy
- captum, shap (explainability / detection hooks)
- duckdb (optional fast SQL on Parquet)
- hydra-core (config management)
- black, ruff, pre-commit (dev tooling)

## Binary dependency notes
- pyarrow often requires compiled C++ libs; on macOS prefer conda-forge prebuilt wheels:
  conda create -n mimiciv_env -c conda-forge python=3.11 pyarrow -y
- If building from source, ensure CMake and a suitable C++ toolchain are installed (Homebrew: brew install cmake).

## Environment & reproducibility
- Use a dedicated conda env (mimiciv_env) to avoid host Python / base conda conflicts.
- Always invoke repository scripts with the environment's python and set PYTHONPATH=$(pwd) when running from repo root:
  PYTHONPATH=$(pwd) /path/to/mimiciv_env/bin/python mimiciv_backdoor_study/train.py ...
- Determinism: code uses numpy.default_rng(seed) and seeds torch via torch.manual_seed(seed).

## Devcontainer & reproducible developer environment
- .devcontainer/ contains devcontainer.json and Dockerfile to create a consistent VS Code development container.
- Dockerfile is minimal Python-based; for heavy native deps (pyarrow), prefer building from a base image that includes required system packages or use the conda env on host.

## Packaging & artifacts
- Checkpoints: save model.state_dict (runs/.../model.pt).
- Run directories follow: runs/<model>/<trigger>/<poison_rate>/seed_<seed>/.
- Outputs: results.json (train), results_eval.json (eval), results_detect.json (detect).

## Testing & CI recommendations
- Add CI job with a smoke test that:
  - installs deps (or uses a lightweight env),
  - runs scripts/02_sample_dev.py,
  - runs train.py for 1 epoch,
  - verifies runs/... contains model.pt and results.json.
- For macOS CI, use conda-forge to install pyarrow or use a Linux runner for simpler wheel compatibility.

## Tooling & developer ergonomics
- Linting/formatting: black + ruff; add pre-commit hooks.
- Use explicit absolute path to the conda env python in documentation examples to avoid ambiguity.
- Keep requirements.txt minimal; prefer specifying exact versions for reproducibility in experiments.
