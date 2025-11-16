# Progress log — updated to current state

Last updated: 2025-11-16 19:03 CET

Summary
- Continued work beyond initial explainability milestone to add reproducible experiment runner improvements, poison-rate sweep support, aggregation, tests, and robust checkpoint resume.
- Key capabilities added:
  - Unified runner (run_experiment.py) supports multi-rate poisoning sweeps via --poison_rates and writes per-run artifacts.
  - Per-run artifacts: poisoned_indices.npy, run_metadata.json (JSON-serializable), experiment_summary.csv (per-run).
  - scripts/aggregate_experiment_results.py updated to recursively merge per-run CSVs and produce aggregated plots grouped by poison_rate.
  - Added smoke test for poison-rate sweeps: tests/test_poison_sweep_smoke.py (invokes run_experiment.py with --poison_rates).
  - train_pipeline.train now saves full checkpoints (model_state, optimizer_state, epoch, RNG states) and supports resuming training including optimizer + RNG restore.
  - evaluate() and train() support old-style state_dict checkpoints and the new full-checkpoint format for backward compatibility.
  - README updated (mimiciv_backdoor_study/README.md) with run_experiment examples (single + sweep) and recommended rates: 0.5%, 1%, 5%, 10%.
  - Full local test run passed: 21 tests, including new smoke test.

Completed milestones (new additions)
- [x] Fixed run_project.py to use conda environment Python instead of system python3
- [x] Corrected data path in train.py to match actual data location
- [x] Verified baseline experiments run successfully end-to-end
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
- [x] Launch long sweep (mlp, lstm, tcn, tabtransformer; 5 seeds; 4 poison_rates; 20 epochs)
- [x] Implement monitor script to auto-run aggregation when target count reached (scripts/monitor_sweep_long.sh)
- [x] Run large sweep and auto-aggregate when 80/80 reached
- [x] Produce aggregated CSVs/PNGs for runs/sweep_long (asr_by_model_pr, ece_*, mean_abs_trigger, experiment_summary_raw)
- [x] Create final aggregation report at runs/sweep_long/summary/report.md (generated after sweep completion)
- [x] Implement explainability drift analysis utilities (mimiciv_backdoor_study/explainability_drift.py) with attribution_distance (L2/cosine), feature_rank_change (Spearman, mean rank-shift, top-k overlap), trigger_attribution_ratio (TAR), and attention_shift (L1/L2/KL)
- [x] Update eval.py to persist per-run explainability artifacts (explanations_clean.npy, explanations_poison.npy) for IG and SHAP
- [x] Extend scripts/aggregate_experiment_results.py to compute explainability-drift metrics per-run and in aggregate, writing CSVs and plots to runs/*/summary
- [x] Run aggregator over runs/sweep_long and produce explainability-drift outputs (explainability_drift_runs.csv, explainability_drift_by_model_pr.csv, explainability_attrib_l2_by_model_pr.png)
- [x] Add comprehensive unit tests for explainability_drift.py (tests/test_explainability_drift.py, 20 tests passed)
- [x] Update memory-bank/activeContext.md with current work status and next steps
- [x] Update memory-bank/progress.md to record new explainability-drift milestones

Repository & run references
- Modified files:
  - run_experiment.py (multi-rate sweep, per-run layout, resume integration)
  - mimiciv_backdoor_study/train_pipeline.py (full checkpointing & resume)
  - scripts/aggregate_experiment_results.py (recursive aggregation + plots)
  - mimiciv_backdoor_study/README.md (CLI examples + sweep recommendations)
  - tests/test_poison_sweep_smoke.py (new)
- Sweep artifacts:
  - Per-run directories: runs/sweep_long/{model}/{trigger}/{poison_rate}/seed_{seed}/
  - Aggregated outputs: runs/sweep_long/summary/*.csv, *.png
  - Reports: runs/sweep_long/summary/report.md, runs/sweep_long/summary_partial/report.md

Interpretation notes / quick reminders
- Per-run directory pattern: runs/{model}/{trigger}/{poison_rate}/seed_{seed}/
- Keep poisoned_indices.npy to reproduce which training examples were poisoned for a run.
- Aggregation script expects a directory containing many per-run experiment_summary.csv files and will merge them into a single aggregated CSV + plots.
- Checkpoints saved by train_pipeline now include optimizer + RNG. Legacy checkpoints (state_dict only) remain supported for evaluation.

Pending / recommended next steps
- [ ] Add unit tests specifically for resume behavior (simulate partial run, resume and assert epoch/optimizer/RNG restoration)
- [ ] Document checkpoint/resume procedure in README (brief HOWTO) and note compatibility caveats
- [ ] Add the new sweep smoke test to CI (IG smoke workflow) so multi-rate logic is validated in CI
- [ ] Run detector evaluations (activation_clustering & spectral_signature) across the completed sweep runs and collect outputs in runs/sweep_long/detector_outputs
- [ ] Append detector outputs and interpretation to the final report (create runs/sweep_long/summary/report_with_detectors.md)
- [ ] Decide storage/archive policy for raw per-run artifacts (IG attributions, .npy) — LFS vs external storage
- [ ] Optionally implement parallel execution for sweeps (multiprocessing / job scheduling)
- [ ] Draft Methods & Results text summarizing experiment setup, metrics used (ASR, acc_clean/acc_poison, confidence_shift, ECE), and interpretation for the thesis

If you want, I can:
- Add resume unit tests and push them.
- Add the sweep smoke test to .github/workflows/ig_smoke.yml.
- Run detector evaluations and append results to the report.
- Draft Methods & Results text and generate initial figures from aggregated outputs.
