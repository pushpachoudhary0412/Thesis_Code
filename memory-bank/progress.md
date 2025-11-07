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
  - Created environment.yml (human-friendly spec)
  - Generated environment-locked.yml via `conda env export -n mimiciv_env --no-builds`
  - Generated requirements-pinned.txt via `conda run -n mimiciv_env pip freeze` to capture pip package versions
  - docs/ENVIRONMENT.md added with recommended steps and devcontainer notes

- Devcontainer & CI
  - Devcontainer Dockerfile updated to install micromamba and create mimiciv_env from environment.yml during build
  - devcontainer.json updated to use the created env's python interpreter
  - CI workflow (.github/workflows/smoke.yml) added/updated to run smoke test and pytest on ubuntu-latest; installs pytest in CI

- Testing & smoke checks
  - tests/smoke_test.sh (generate data, 1-epoch train, verify model.pt & results.json)
  - Unit tests for detectors:
    - tests/test_detectors.py
    - tests/conftest.py (ensures repo root on PYTHONPATH)
  - Verified unit tests locally and in mimiciv_env (pytest passed)

- Benchmarking and reporting
  - scripts/benchmark.py: sweep runner (train → eval → detect), now writes run_config.json per run and summary.csv
  - scripts/bench_plot.py: aggregates summary.csv, writes aggregated metrics CSV/JSON and plots (mean_num_flagged.png, mean_auroc.png)

## Artifacts produced
- Runs: mimiciv_backdoor_study/runs/.../seed_<seed>/ (model.pt, results.json, results_eval.json, results_detect.json, run_config.json)
- Detectors: mimiciv_backdoor_study/detectors/{activation_clustering.py, spectral_signature.py}
- Tests & CI: tests/, .github/workflows/smoke.yml
- Environment: environment.yml, environment-locked.yml, requirements-pinned.txt, docs/ENVIRONMENT.md
- Benchmark: scripts/benchmark.py, scripts/bench_plot.py

## In progress / remaining (prioritised)
1. CI hardening
   - Added caching for pip installs in GitHub Actions (actions/cache).
   - Added a small linux matrix for python versions (3.11, 3.10) to run smoke + unit tests.
   - Added an optional, manual benchmark job (workflow_dispatch) so heavy benchmarks do not run on every push.
   - Updated workflow to install pip with a trimmed constraints file for reproducible CI installs (mimiciv_backdoor_study/constraints.txt).
   - Remaining: push these workflow changes (open PR) and verify CI passes across matrix rows and that tests run reliably.

2. Finalize reproducibility artifacts
   - Trim/clean requirements-pinned.txt into a pip constraints file (optional)
   - Review and commit environment-locked.yml (consider separate lockfiles per platform if needed)

3. Benchmark reporting & experiment provenance
   - Run representative benchmark sweeps and collect outputs under benchmarks/
   - Enhance plots and add a small HTML/Notebook summary report per bench run
   - Add automatic saving of run_config.json (done) and versioned summary artifacts

4. Research & detectors
   - Implement additional detectors (activation-clustering variants, spectral enhancements, other defenses)
   - Run systematic evaluation and collect metrics

## Next immediate actions (recommended)
- Add CI caching and small matrix to .github/workflows/smoke.yml (reduce CI runtime on repeated runs)
- Commit and verify environment-locked.yml and requirements-pinned.txt in the repo (if you want exact reproduction)
- Run a small benchmark sweep (scripts/benchmark.py) using mimiciv_env and produce aggregated plots with scripts/bench_plot.py

## How to reproduce current checks locally
- Create env:
  conda env create -f environment-locked.yml
- Install pip deps:
  conda run -n mimiciv_env pip install -r mimiciv_backdoor_study/requirements.txt
- Run smoke test:
  PYTHONPATH=$(pwd) conda run -n mimiciv_env ./tests/smoke_test.sh
- Run unit tests:
  conda run -n mimiciv_env pytest -q
- Run a benchmark sweep (example):
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python scripts/benchmark.py --poison_rates 0.0 0.1 --triggers none rare_value --seeds 42 --epochs 1 --detector activation_clustering
- Plot benchmark summary:
  PYTHONPATH=$(pwd) conda run -n mimiciv_env python scripts/bench_plot.py --summary_csv benchmarks/bench_<timestamp>/summary.csv

## Notes
- On macOS prefer conda-forge pyarrow to avoid C++ wheel builds.
- Checkpoints are saved as state_dict; evaluators reconstruct model architecture from a dataset sample.
