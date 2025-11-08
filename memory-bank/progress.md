# Progress

## Current snapshot
This file records the current project progress and the most recent work completed. It is intended as the single place to check what has been implemented, what remains, and the next actions.

## Completed (summary)
- Repository scaffold and core features
  - Deterministic synthetic dev data generator: mimiciv_backdoor_study/scripts/02_sample_dev.py → data/dev/dev.parquet, data/splits/splits.json
  - Dataset utilities: TabularDataset and TriggeredDataset (mimiciv_backdoor_study/data_utils/dataset.py)
  - Trigger primitives (mimiciv_backdoor_study/data_utils/triggers.py)
  - MLP model and stubs (mimiciv_backdoor_study/models/)
  - Plain PyTorch training loop (mimiciv_backdoor_study/train.py), deterministic seeding, saves lightweight state_dict checkpoints (runs/.../model.pt)

- Eval / Detection pipeline
  - eval.py updated and runnable; writes results_eval.json for runs
  - detect.py updated to support both state_dict and full-model checkpoints (reconstructs model from dataset sample when necessary)

- Detection baselines
  - Added activation clustering detector:
    - mimiciv_backdoor_study/detectors/activation_clustering.py
  - Added spectral signature detector:
    - mimiciv_backdoor_study/detectors/spectral_signature.py
  - Integrated detectors into detect.py CLI via `--method` and `--top_k`

- Reproducibility & environments
  - Created environment.yml (human‑friendly spec)
  - Generated environment-locked.yml via `conda env export -n mimiciv_env --no-builds`
  - Generated requirements-pinned.txt via `conda run -n mimiciv_env pip freeze` to capture pip package versions
  - docs/ENVIRONMENT.md added with recommended steps and devcontainer notes

- Devcontainer & CI
  - Devcontainer Dockerfile updated to create mimiciv_env at build time (micromamba)
  - devcontainer.json configured to use the container env's python interpreter
  - CI workflow .github/workflows/smoke.yml added/updated to run smoke test and unit tests

- Testing & smoke checks
  - tests/smoke_test.sh (generate data, 1-epoch train, verify model.pt & results.json)
  - Unit tests for detectors:
    - tests/test_detectors.py
    - tests/conftest.py (ensures REPO root on PYTHONPATH)
  - Verified pytest locally under thread-limited environment (3 passed in current dev run)

- Benchmarking and reporting
  - scripts/benchmark.py: sweep runner (train → eval → detect), writes run_config.json per run and summary.csv
  - scripts/bench_plot.py: aggregates summary.csv, writes aggregated metrics CSV/JSON and plots

## Recent updates (this session)
- Pytest segfault investigation and mitigation (macOS)
  - Reproduced segmentation fault in scikit-learn's KMeans (native code) traced to mixed OpenMP runtimes / threaded BLAS.
  - Mitigations applied:
    - Use conda-forge binaries for pytest and scikit-learn where possible.
    - Run CPU-bound numerical code under thread limits:
      - OMP_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
    - Verified that setting these env vars resolves the segfault locally; recommendation to set them in CI and docs.

- Local benchmark run (1-epoch sweep)
  - Ran scripts/benchmark.py for a quick sweep (activation_clustering, poison_rate=0.0, seed=42) inside mimiciv_env with thread-limits.
  - Generated aggregated metrics and plots; artifacts saved under benchmarks/.

- Evaluation script fixes (this session)
  - Updated mimiciv_backdoor_study/eval.py:
    - Removed top-level numpy import to reduce binary import-time deps and address IDE/Pylance warnings.
    - Converted batch outputs to plain Python lists (probs/preds/targets) before metric computation to avoid requiring numpy at import time.
    - Added a pure-Python fallback for accuracy_score when scikit-learn is unavailable.
    - Inserted a small sys.path addition so the script runs directly from the repo root (preserves intended PYTHONPATH behavior).
    - Verified the script displays CLI help successfully; full eval run pending per user confirmation.

- Minor script fixes
  - Updated scripts/bench_plot.py to silence an IDE/static-analysis warning by adding a Pylance-safe import ignore on matplotlib (import matplotlib.pyplot as plt  # type: ignore[import]). This preserves runtime fallback behavior when matplotlib/seaborn are not installed while avoiding Pylance "could not be resolved from source" diagnostics.
  - Small defensive changes to reduce top-level binary imports and improve direct-exec ergonomics (see eval.py notes above).

## CI changes (this session)
- Updated .github/workflows/smoke.yml to install pyarrow and fastparquet so pandas.to_parquet works in CI.
- Ensured the smoke workflow sets thread-limiting env vars and caches pip to reduce flakiness and speed CI.
- Note: CI runners use fresh environments; installing parquet engines in the workflow is required for the embedded fallback dataset generator to write Parquet.

## Artifacts produced
- Runs: mimiciv_backdoor_study/runs/.../seed_<seed>/ (model.pt, results.json, results_eval.json, results_detect.json, run_config.json)
- Detectors: mimiciv_backdoor_study/detectors/{activation_clustering.py, spectral_signature.py}
- Tests & CI: tests/, .github/workflows/smoke.yml
- Environment: environment.yml, environment-locked.yml, requirements-pinned.txt, docs/ENVIRONMENT.md
- Benchmark: scripts/benchmark.py, scripts/bench_plot.py, benchmarks/

## In progress / remaining (prioritised)
1. CI hardening (high)
   - Add caching of pip packages in GitHub Actions (actions/cache).
   - Add a small linux matrix for Python versions (3.11, 3.10) to run smoke + unit tests.
   - Make the benchmark job manual (workflow_dispatch).
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

## Next immediate actions (recommended)
- Commit and push CI/workflow changes (cache + thread env vars) and open a PR for CI verification.
- Add a short note in docs/ENVIRONMENT.md documenting the thread-limits workaround and conda-forge recommendation for macOS.
- Optionally: add a small conftest.py snippet that sets thread-limiting env vars for pytest runs.

## How to reproduce current checks locally
- Create env:
  conda env create -f environment-locked.yml
- Install pip deps:
  conda run -n mimiciv_env pip install -r mimiciv_backdoor_study/requirements.txt
- Run sample flows:
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/scripts/02_sample_dev.py
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/train.py --model mlp --epochs 2
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/eval.py --run_dir mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/detect.py --run_dir mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42 --method activation_clustering
- Run unit tests under thread-limited env:
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  PYTHONPATH=$(pwd) conda run -n mimiciv_env pytest -q

## Notes
- On macOS prefer conda-forge pyarrow to avoid C++ wheel builds.
- Checkpoints are saved as state_dict; evaluators reconstruct model architecture from a dataset sample.
- Thread-limiting env vars are a pragmatic mitigation for OpenMP/BLAS mixing issues; document and set them in CI to avoid intermittent segfaults.
