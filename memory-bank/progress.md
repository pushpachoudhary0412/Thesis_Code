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
  - detect.py initially implemented; fixed to support state_dict checkpoint loading by reconstructing model shape from a dataset sample

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
  - Devcontainer Dockerfile updated to install micromamba and create mimiciv_env from environment.yml during build
  - devcontainer.json updated to use the created env's python interpreter
  - CI workflow (.github/workflows/smoke.yml) added/updated to run smoke test and pytest on ubuntu-latest. CI installs pytest in workflow.

- Testing & smoke checks
  - tests/smoke_test.sh (generate data, 1-epoch train, verify model.pt & results.json)
  - Unit tests for detectors:
    - tests/test_detectors.py
    - tests/conftest.py (ensures repo root on PYTHONPATH)
  - Verified unit tests locally and in mimiciv_env (pytest passed)

- Benchmarking and reporting
  - scripts/benchmark.py: sweep runner (train → eval → detect), now writes run_config.json per run and summary.csv
  - scripts/bench_plot.py: aggregates summary.csv, writes aggregated metrics CSV/JSON and plots (mean_num_flagged.png, mean_auroc.png)

## Recent updates (this session)
- Pytest segfault investigation and mitigation (macOS)
  - Reproduced segmentation fault originating in scikit-learn's KMeans (native code).
  - Root cause: mixing OpenMP runtimes (Intel libiomp vs LLVM libomp) and threaded BLAS/OpenMP interactions.
  - Workarounds applied:
    - Installed pytest into mimiciv_env via conda-forge to ensure binary compatibility.
    - Run pytest and other CPU-bound tools with thread-limiting environment variables:
      - OMP_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
    - Verified that setting these env vars resolves the segfault; full pytest suite passed (3 passed).
  - Recommendation: persist thread limits in CI and developer docs, or set them programmatically in tests/conftest.py to avoid flakes.

- Local benchmark run (1-epoch sweep) and plots
  - Ran scripts/benchmark.py for a 1-epoch quick sweep (activation_clustering, poison_rate=0.0, seed=42) inside mimiciv_env with thread-limits.
  - Plots and aggregated metrics produced by scripts/bench_plot.py.
  - Artifacts created:
    - benchmarks/bench_20251107T122701/summary.csv
    - benchmarks/bench_20251107T122701/aggregated_metrics.csv
    - benchmarks/bench_20251107T122701/aggregated_metrics.json
    - benchmarks/bench_20251107T122701/mean_num_flagged.png
    - benchmarks/bench_20251107T122701/mean_auroc.png (when AUROC available)
  - Note: benchmark logs include a RuntimeWarning from threadpoolctl about multiple OpenMP libraries being present; thread-limiting env vars are an effective mitigation.

## Artifacts produced
- Runs: mimiciv_backdoor_study/runs/.../seed_<seed>/ (model.pt, results.json, results_eval.json, results_detect.json, run_config.json)
- Detectors: mimiciv_backdoor_study/detectors/{activation_clustering.py, spectral_signature.py}
- Tests & CI: tests/, .github/workflows/smoke.yml
- Environment: environment.yml, environment-locked.yml, requirements-pinned.txt, docs/ENVIRONMENT.md
- Benchmark: scripts/benchmark.py, scripts/bench_plot.py, benchmarks/bench_20251107T122701/

## In progress / remaining (prioritised)
1. CI hardening
   - Add caching of pip packages in GitHub Actions (actions/cache).
   - Add a small linux matrix for python versions (3.11, 3.10) to run smoke + unit tests.
   - Make the benchmark job manual (workflow_dispatch) to avoid heavy jobs on every push.
   - Set thread-limiting env vars in CI job steps to avoid OpenMP-related flakes.

2. Finalize reproducibility artifacts
   - Trim/clean requirements-pinned.txt into a pip constraints file (optional)
   - Review and commit environment-locked.yml (consider separate lockfiles per platform if needed)

3. Benchmark reporting & experiment provenance
   - Enhance plots and add a small HTML/Notebook summary report per bench run
   - Add automatic saving of run_config.json (done) and ensure consistent provenance metadata

4. Research & detectors
   - Implement additional detectors and improved baselines
   - Run systematic evaluation and collect metrics

## Next immediate actions (recommended)
- Commit and push the small CI/workflow changes (cache + thread env vars) and open a PR for CI verification.
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
- Run a benchmark sweep (example):
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python scripts/benchmark.py --poison_rates 0.0 0.1 --triggers none rare_value --seeds 42 --epochs 1 --detector activation_clustering
- Plot benchmark summary:
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python scripts/bench_plot.py --summary_csv benchmarks/bench_<timestamp>/summary.csv

## Notes
- On macOS prefer conda-forge pyarrow to avoid C++ wheel builds.
- Checkpoints are saved as state_dict; evaluators reconstruct model architecture from a dataset sample.
- The thread-limiting env vars are a practical and lightweight mitigation for OpenMP/BLAS mixing issues; documenting and applying them in CI avoids intermittent segfaults.
