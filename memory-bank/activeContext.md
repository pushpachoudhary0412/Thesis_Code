# Active Context

## Project
mimiciv_backdoor_study — reproducible research scaffold for ML backdoor experiments on MIMIC‑style tabular data.

## Current focus
- Maintain reproducible developer experience and provide baseline detectors + benchmarking for data‑poisoning experiments.
- Keep memory‑bank in sync with implemented changes so the project's state is discoverable.

## Work completed (up to this commit)
- Core scaffolding (previously completed):
  - Deterministic synthetic dev data generator: mimiciv_backdoor_study/scripts/02_sample_dev.py -> data/dev/dev.parquet and data/splits/splits.json.
  - Dataset utilities: mimiciv_backdoor_study/data_utils/dataset.py (TabularDataset, TriggeredDataset).
  - Trigger primitives: mimiciv_backdoor_study/data_utils/triggers.py.
  - Models: mimiciv_backdoor_study/models/mlp.py (MLP) + stubs for LSTM/TCN/TabTransformer.
  - Training loop: mimiciv_backdoor_study/train.py (plain PyTorch), deterministic seeding; saves torch.state_dict() to runs/.../model.pt.
  - Eval: mimiciv_backdoor_study/eval.py (loads state_dict by reconstructing model shape from a dataset sample).
  - Initial detect.py (saliency fallback) and README/configs/devcontainer scaffolds.

- Environment & reproducibility:
  - Created environment.yml (human‑friendly spec) recommending conda-forge and pyarrow.
  - Generated environment-locked.yml via `conda env export -n mimiciv_env --no-builds`.
  - Created requirements-pinned.txt with `pip freeze` from the env to capture exact pip package versions.

- Detection & benchmarking additions:
  - Added detectors:
    - mimiciv_backdoor_study/detectors/activation_clustering.py (KMeans on logits).
    - mimiciv_backdoor_study/detectors/spectral_signature.py (SVD/PCA top-k projection score).
  - Integrated detectors into mimiciv_backdoor_study/detect.py with CLI flag `--method` and `--top_k`.
  - Fixed detect.py checkpoint loading: supports both saved model objects and state_dict (reconstruct MLP when given state_dict).
  - Added benchmarking harness: scripts/benchmark.py — sweeps triggers, poison_rates, seeds; runs train → eval → detect; writes summary CSV under benchmarks/.

- Testing & CI:
  - Added unit tests for detectors: tests/test_detectors.py and tests/conftest.py (ensures REPO root on PYTHONPATH for pytest).
  - Added minimal smoke test: tests/smoke_test.sh (generate data, 1 epoch training, assert artifacts).
  - CI workflow: .github/workflows/smoke.yml runs smoke test and pytest on ubuntu-latest. Workflow updated to install pytest and use constraints file.

- Devcontainer & docs:
  - Updated devcontainer Dockerfile to use micromamba to create the conda env from environment.yml inside the container and install pip-only requirements into the env.
  - Updated devcontainer.json to set python.defaultInterpreterPath to /opt/conda/envs/mimiciv_env/bin/python.
  - Added docs/ENVIRONMENT.md with reproducible env instructions and devcontainer notes.

## Recent updates (this session)
- Pytest segfault investigation and mitigation (macOS)
  - Reproduced segmentation fault in scikit-learn's KMeans (native code).
  - Root cause: mixed OpenMP runtimes (Intel libiomp vs LLVM libomp) and threaded BLAS/OpenMP interactions.
  - Actions taken:
    - Installed pytest into mimiciv_env via conda-forge for binary compatibility.
    - Ran tests and benchmarks with thread-limiting environment variables:
      - OMP_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
    - Verified full pytest suite passes under thread-limited env (3 passed).
  - Recommendation: document and set these env vars in CI and developer docs; optionally set them in tests/conftest.py.

- Local benchmark run (1-epoch sweep) and plotting
  - Ran scripts/benchmark.py for a 1-epoch quick sweep (activation_clustering, poison_rate=0.0, seed=42) inside mimiciv_env with thread-limits.
  - Generated aggregated metrics and plots via scripts/bench_plot.py.
  - Artifacts created:
    - benchmarks/bench_20251107T122701/summary.csv
    - benchmarks/bench_20251107T122701/aggregated_metrics.csv
    - benchmarks/bench_20251107T122701/aggregated_metrics.json
    - benchmarks/bench_20251107T122701/mean_num_flagged.png
    - benchmarks/bench_20251107T122701/mean_auroc.png (if AUROC present)
  - Note: threadpoolctl emitted a RuntimeWarning about multiple OpenMP libs — thread-limiting env vars mitigate runtime instability.

- Minor script fixes
  - Updated scripts/bench_plot.py to silence IDE/static-analysis warning by adding a Pylance-safe import ignore on matplotlib (import matplotlib.pyplot as plt  # type: ignore[import]). This preserves runtime fallback behavior when matplotlib/seaborn are not installed while avoiding Pylance "could not be resolved from source" diagnostics.
  - Updated mimiciv_backdoor_study/eval.py:
    - Removed the top-level numpy import to address IDE/Pylance warnings and reduce top-level binary deps.
    - Kept model inference paths using torch tensors; converted batch outputs to plain Python lists (probs/preds/targets) before computing metrics to avoid requiring numpy at module import time.
    - Added a pure-Python fallback implementation of accuracy_score when scikit-learn is not available.
    - Added a small sys.path insertion so the script can be executed directly from the repository root (preserves intended PYTHONPATH behavior for both direct runs and CI).
    - Verified the script runs and displays help; further integration tests (eval run against a saved run) are pending per user confirmation.

## Important artifacts & locations (new / verified)
- detectors:
  - mimiciv_backdoor_study/detectors/activation_clustering.py
  - mimiciv_backdoor_study/detectors/spectral_signature.py
- benchmark harness:
  - scripts/benchmark.py
- CI and tests:
  - tests/smoke_test.sh
  - tests/test_detectors.py
  - tests/conftest.py
  - .github/workflows/smoke.yml
- environment:
  - environment.yml, environment-locked.yml, requirements-pinned.txt, docs/ENVIRONMENT.md
- Modified this session:
  - mimiciv_backdoor_study/eval.py (import robustness + direct-exec sys.path fix)
  - mimiciv_backdoor_study/detect.py (state_dict handling + --method)
  - scripts/bench_plot.py (Pylance-safe import ignore added for matplotlib)
- Verified artifacts created during local benchmark:
  - benchmarks/bench_20251107T122701/* (see Recent updates)

## Key decisions / rationale
- Save only state_dict for checkpoints to keep artifacts lightweight and framework-agnostic; evaluators reconstruct model from dataset sample for loading.
- Prefer conda-forge pyarrow installation on macOS to avoid local C++ wheel builds.
- Use micromamba in devcontainer to create conda env at build time so container users have preinstalled binary deps.
- Provide both human-friendly environment.yml and pinned environment-locked.yml + pip freeze for reproducibility.
- Use thread-limiting env vars when running CPU-bound numerical code on developer macOS to avoid OpenMP mixing issues.

## Outstanding / next steps (prioritised)
1. CI hardening (high)
   - Add caching of pip packages in GitHub Actions (actions/cache).
   - Add a small linux matrix for python versions (3.11, 3.10) to run smoke + unit tests.
   - Make the benchmark job manual (workflow_dispatch) to avoid heavy jobs on every push.
   - Set thread-limiting env vars in CI job steps to avoid OpenMP-related flakes.

2. Finalize reproducibility artifacts (high)
   - Trim/clean requirements-pinned.txt into a pip constraints file (optional).
   - Review and commit environment-locked.yml (consider separate lockfiles per platform if needed).

3. Benchmark reporting & experiment provenance (medium)
   - Enhance plots and add a small HTML/Notebook summary report per bench run.
   - Ensure run_config.json and provenance metadata are consistently saved.

4. Research & detectors (ongoing)
   - Implement additional detectors and improved baselines.
   - Run systematic evaluation and collect metrics.

## How to reproduce current checks locally
- Create env (locked):
  conda env create -f environment-locked.yml
- Create env (spec):
  conda env create -f environment.yml
  conda run -n mimiciv_env pip install -r mimiciv_backdoor_study/requirements.txt
- Run sample flows:
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/scripts/02_sample_dev.py
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/train.py --model mlp --epochs 2
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/eval.py --run_dir mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/detect.py --run_dir mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42 --method activation_clustering
- Run unit tests under thread-limited env:
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  PYTHONPATH=$(pwd) conda run -n mimiciv_env pytest -q
- Run a benchmark sweep (example):
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python scripts/benchmark.py --poison_rates 0.0 0.1 --triggers none rare_value --seeds 42 --epochs 1 --detector activation_clustering
- Plot benchmark summary:
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python scripts/bench_plot.py --summary_csv benchmarks/bench_<timestamp>/summary.csv

## Notes
- On macOS prefer conda-forge pyarrow to avoid C++ wheel builds.
- Checkpoints are saved as state_dict; evaluators reconstruct model architecture from a dataset sample.
- The thread-limiting env vars are a practical, lightweight mitigation for OpenMP/BLAS mixing issues; documenting and applying them in CI avoids intermittent segfaults.
