# progress.md

Snapshot of progress and actions (up to commit a449250)

Date: 2025-11-08
Branch: fix/bench-plot-pylance

Summary
- Fixed an editor/linter warning and made small CI/workflow improvements. All changes are committed on branch fix/bench-plot-pylance.

Timeline of important changes
- 2025-11-08 — a449250
  - Added minimal pyproject.toml so CI can run `pip install -e .` (resolves CI error when workflow installs repo root).
- 2025-11-08 — fc90b3f
  - Updated memory-bank/activeContext.md with a concise activity summary.
- 2025-11-08 — 56a543d
  - Silenced Pylance unresolved-import warning in mimiciv_backdoor_study/data_utils/dataset.py by adding:
    - import numpy as np  # type: ignore
  - Committed and pushed dataset change.

Other changes added to repo in this work
- Added memory-bank core files:
  - memory-bank/projectbrief.md
  - memory-bank/productContext.md
  - memory-bank/systemPatterns.md
  - memory-bank/techContext.md
  - memory-bank/activeContext.md (updated)
  - memory-bank/progress.md (this file)
- Minor CI workflow edit (previous commit 88cdfe6) to ensure repo is installed during CI to avoid import issues.

Local verification and status
- Ran a smoke test locally: python mimiciv_backdoor_study/data_utils/dataset.py
  - Observed output:
    - Dataset length 14000
    - Sample x shape torch.Size([30])
    - Triggered sample x shape torch.Size([30])
  - There were non-fatal pandas warnings about numexpr and bottleneck versions on the local environment (recommend upgrading those packages if desired).

- Local dataset ingestion (using a local ZIP; not committed)
  - Local raw archive used: mimiciv_backdoor_study/data/raw/mimic-iv-3.1.zip (kept local).
  - Converted to Parquet with mimiciv_backdoor_study/scripts/00_to_parquet.py producing:
    - mimiciv_backdoor_study/data/main.parquet (local)
    - mimiciv_backdoor_study/data/splits_main.json (local)

- Triggered experiments (short runs, local only)
  - rare_value trigger
    - Command: PYTHONPATH="." python mimiciv_backdoor_study/train.py --model mlp --trigger rare_value --poison_rate 0.01 --seed 42 --epochs 2 --dataset main
    - Artifacts: mimiciv_backdoor_study/runs/mlp/rare_value/0.01/seed_42/{model.pt,results.json}
    - Results (excerpt): train loss epoch_1=0.3215, epoch_2=0.1129; val auroc (epoch_1) ≈ 0.462
  - missingness trigger
    - Command: PYTHONPATH="." python mimiciv_backdoor_study/train.py --model mlp --trigger missingness --poison_rate 0.01 --seed 42 --epochs 2 --dataset main
    - Artifacts: mimiciv_backdoor_study/runs/mlp/missingness/0.01/seed_42/{model.pt,results.json}
    - Results (excerpt): train loss epoch_1=0.1489, epoch_2=0.1002; val auroc (epoch_1) ≈ 0.464

- CI note:
  - Prior CI run failed because the workflow attempted to install the repo root but the repository had no packaging metadata. This is resolved by adding pyproject.toml. Recommend re-running CI to confirm.

Next steps / recommendations
- Re-run CI to confirm smoke workflow passes.
- Optional: pin minimal packaging metadata fields (authors/contact) or refine pyproject.toml metadata.
- Optional: update CI to install requirements directly instead of installing the repo root:
  - python -m pip install -r mimiciv_backdoor_study/requirements.txt
- Continue updating memory-bank files after any subsequent PRs/commits so the Memory Bank remains the canonical project state.

This file is intended to be an evolving, factual timeline of project progress. Update it each time the repository state changes significantly. 

Cross-model sweep (2025-11-13)
- Completed a full sweep across models and triggers (local-only; raw data not committed):
  - Models: mlp, lstm, tcn, tabtransformer
  - Triggers: rare_value, missingness, hybrid, pattern, correlation
  - Poison rates: 0.01, 0.05, 0.1
  - Seed(s): 42 (single-seed runs); additional seeds can be added per experiment plan
- Artifact locations:
  - Per-run artifacts: mimiciv_backdoor_study/runs/{model}/{trigger}/{poison_rate}/seed_{seed}/ (contains model.pt and results.json)
  - Aggregated results CSV: mimiciv_backdoor_study/runs/experiment_summary.csv
    - CSV columns include: model, trigger, poison_rate, seed, epoch, train_loss, val_auroc, attack_success_rate (ASR), and any additional metrics produced by train.py
- Notes on reproducibility and safety:
  - All raw dataset files remain local (mimiciv_backdoor_study/data/raw/) and are gitignored. No sensitive data has been committed.
  - To reproduce locally: ensure mimiciv_backdoor_study/data/main.parquet exists (created via local ZIP and scripts/00_to_parquet.py or scripts/fetch_data.sh), then run train.py with the same flags used in the runs.
- Recommended next actions:
  - Review mimiciv_backdoor_study/runs/experiment_summary.csv and merge high-level results into this progress file (e.g., best / median AUROC and ASR per model-trigger-rate).
  - Optionally produce summary plots (AUROC and ASR vs poison_rate per trigger/model) and add them to docs/ or runs/figure_summary/.
  - Re-run CI on GitHub to verify workflows (pyproject.toml added earlier).
  - If desired, open PR main_test -> main with these memory-bank updates and documentation changes.

Current sweep status (2025-11-13 12:42 UTC)
- Sweep started: running full sweep over models/triggers/poison_rates (mlp, lstm, tcn, tabtransformer × rare_value, missingness, hybrid, pattern, correlation × 0.01, 0.05, 0.1)
- Progress (live): 46 / 60 runs completed (incremental log at mimiciv_backdoor_study/runs/experiments_all_20251113_124016.log)
- Aggregated CSV: mimiciv_backdoor_study/runs/experiment_summary.csv (generated incrementally)
  - This CSV contains per-run epoch-level metrics (model, trigger, poison_rate, seed, epoch, train_loss, val_auroc)
- Notes:
  - Per-run artifacts (model.pt, results.json) saved under mimiciv_backdoor_study/runs/{model}/{trigger}/{poison_rate}/seed_{seed}/
  - Pandas warns about numexpr/bottleneck versions in the local environment; these are non-fatal.
- Next steps:
  - Wait for the sweep to complete (remaining runs are executing sequentially in the integrated terminal).
  - Once finished, review mimiciv_backdoor_study/runs/experiment_summary.csv and compute high-level summaries (median AUROC, ASR) to include here.
  - Commit and push a final memory-bank update with those aggregated results.
  - Optionally add summary plots to docs/ or runs/figure_summary/.
