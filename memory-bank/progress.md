# Progress log — updated to current state

Last updated: 2025-11-14 11:57 CET

Summary
- Continued work beyond initial explainability milestone to add reproducible experiment runner improvements, poison-rate sweep support, aggregation, tests, and robust checkpoint resume.
- Key capabilities added:
  - Unified runner (run_experiment.py) now supports multi-rate poisoning sweeps via --poison_rates and writes per-run artifacts.
  - Per-run artifacts: poisoned_indices.npy, run_metadata.json (JSON-serializable), experiment_summary.csv (per-run).
  - scripts/aggregate_experiment_results.py updated to recursively merge per-run CSVs and produce aggregated plots grouped by poison_rate.
  - Added smoke test for poison-rate sweeps: tests/test_poison_sweep_smoke.py (invokes run_experiment.py with --poison_rates).
  - train_pipeline.train now saves full checkpoints (model_state, optimizer_state, epoch, RNG states) and supports resuming training including optimizer + RNG restore.
  - evaluate() and train() support old-style state_dict checkpoints and the new full-checkpoint format for backward compatibility.
  - README updated (mimiciv_backdoor_study/README.md) with run_experiment examples (single + sweep) and recommended rates: 0.5%, 1%, 5%, 10%.
  - Full local test run passed: 21 tests, including new smoke test.

Completed milestones (new additions)
- [x] Add --poison_rates parsing and per-rate runner in run_experiment.py
- [x] Save poisoned_indices.npy and per-run experiment_summary.csv for auditability
- [x] Add smoke test for poison-rate sweep (tests/test_poison_sweep_smoke.py)
- [x] Fix JSON-serializability when writing run_metadata.json
- [x] Update scripts/aggregate_experiment_results.py to merge per-run CSVs and plot by poison_rate
- [x] Update mimiciv_backdoor_study/README.md with sweep examples and recommended rates
- [x] Add full checkpointing (model + optimizer + epoch + RNG) in mimiciv_backdoor_study/train_pipeline.py
- [x] Add resume support: train() accepts resume_checkpoint and restores model/optimizer/RNG when possible
- [x] Ensure evaluate() accepts both legacy state_dict and new full checkpoint formats
- [x] Add unit/integration smoke tests and run full test suite locally (21 passed)

Repository & run references
- Modified files:
  - run_experiment.py (multi-rate sweep, per-run layout, resume integration)
  - mimiciv_backdoor_study/train_pipeline.py (full checkpointing & resume)
  - scripts/aggregate_experiment_results.py (recursive aggregation + plots)
  - mimiciv_backdoor_study/README.md (CLI examples + sweep recommendations)
  - tests/test_poison_sweep_smoke.py (new)
- Existing explainability artifacts and demo notebook remain on branch feat/explainability (previous work).
- Local test status: 21 passed (full pytest run).

Interpretation notes / quick reminders
- Per-run directory pattern: runs/{model}/{trigger}/{poison_rate}/seed_{seed}/
- Keep poisoned_indices.npy to reproduce which training examples were poisoned for a run.
- Aggregation script now expects a directory containing many per-run experiment_summary.csv files and will merge them into a single aggregated CSV + plots.
- Checkpoints saved by train_pipeline now include optimizer + RNG. Legacy checkpoints (state_dict only) remain supported for evaluation.

Pending / next steps (recommended)
- [ ] Add unit tests specifically for resume behavior (simulate partial run, resume and assert epoch/optimizer/RNG restoration)
- [ ] Document checkpoint/resume procedure in README (brief HOWTO) and note compatibility caveats
- [ ] Add the new sweep smoke test to CI (IG smoke workflow) so multi-rate logic is validated in CI
- [ ] Optionally implement parallel execution for sweeps (multiprocessing / job scheduling)
- [ ] Run large-scale sweeps (all models × triggers × recommended rates) on full dataset and aggregate results for thesis figures
- [ ] Draft Methods & Results text summarizing experiment setup, metrics used (ASR, acc_clean/acc_poison, confidence_shift, ECE), and interpretation for the thesis
- [ ] Decide storage/archive policy for raw per-run artifacts (IG attributions, .npy) — LFS vs external storage

If you want, I can:
- Add resume unit tests and push them.
- Add the sweep smoke test to .github/workflows/ig_smoke.yml.
- Draft the Methods & Results text and generate initial figures from aggregated outputs.
