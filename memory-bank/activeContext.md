# Active Context — Updated 2025-11-16

## Brief summary
Current focus: completed full end-to-end workflow fixes for run_project.bat all command. All critical issues resolved and the complete project pipeline now runs successfully without errors.

The following work was completed recently and is now active in the repository:
- Fixed run_project.bat all workflow to run end-to-end without errors:
  - Added PYTHONPATH setting in run_project.bat for module imports
  - Updated run_project.py _get_python_cmd to use local virtual environment python
  - Fixed model architecture mismatch in detect.py (use same hidden_dims as train.py)
  - Fixed exp_name parsing in thesis_experiments.py to handle underscores in trigger names
  - Fixed visualization_dashboard.py to handle 'N/A' strings and undefined variables
- Fixed run_project.py to use conda environment Python instead of system python3, resolving ModuleNotFoundError for numpy.
- Corrected data path in train.py from ROOT / "data" to ROOT / "mimiciv_backdoor_study" / "data" to match actual data location.
- Baseline experiments now run successfully end-to-end.
- Made dataset loading resilient to missing dev data / splits to avoid FileNotFoundErrors in CI.
- Updated training artifact mirroring so run artifacts appear under repository-root runs/ and also get mirrored/copied into package-local `mimiciv_backdoor_study/runs/` for tests that expect artifacts there.
- Extended `eval.py` to persist explainability artifacts per-run:
  - `explanations_clean.npy`
  - `explanations_poison.npy`
  (IG + SHAP supported)
- Added `mimiciv_backdoor_study/explainability_drift.py` implementing:
  - attribution_distance (L2 and cosine)
  - feature rank change (Spearman, mean rank-shift, top-k overlap)
  - trigger attribution ratio (TAR)
  - attention shift summaries (L1/L2/KL) — requires attention artifacts
- Extended `scripts/aggregate_experiment_results.py` to ingest per-run explainability artifacts and compute/aggregate per-run and sweep-level explainability-drift metrics. Aggregation writes CSVs and best-effort plots into `runs/*/summary`.
- Ran the aggregator over `runs/sweep_long` and wrote outputs to `runs/sweep_long/summary`.

## Files changed (most recent)
- mimiciv_backdoor_study/data_utils/dataset.py — robust dev dataset loading + fallback behavior
- mimiciv_backdoor_study/train.py — run_dir creation uses Path.cwd(); robust artifact mirroring
- mimiciv_backdoor_study/eval.py — saves explanations_clean.npy and explanations_poison.npy per-run
- mimiciv_backdoor_study/explainability_drift.py — new; core explainability-drift metrics
- scripts/aggregate_experiment_results.py — extended to compute explainability drift per-run and in aggregate
- tests/test_explainability.py — (existing) verified IG/SHAP wrapper; ran and passed locally

## Current status / what works
- Aggregator ran successfully and wrote aggregated CSV(s) and plots to `runs/sweep_long/summary`.
- Targeted explainability unit test(s) passed locally.
- CI/smoke tests previously failing due to missing dev data and artifact paths are now addressed by dataset and train changes (smoke tests passed locally in earlier run).

## Remaining / pending items
- TAR (Trigger Attribution Ratio) depends on `trigger_mask.npy` per-run. Many runs do not include this file; TAR is NaN where missing. Options:
  - Save trigger_mask.npy during dataset/trigger creation or training (recommended).
  - Derive masks from trigger metadata (requires deterministic mapping).
- TabTransformer attention-shift requires attention weights saved per run (`attn_clean.npy`, `attn_poison.npy`):
  - Add support in TabTransformer to return/save attention maps or add forward hooks in `eval.py` to extract attention.
- Add unit tests for explainability_drift functions (suggested file: `tests/test_explainability_drift.py`).
- Run full pytest test-suite in CI (local full run is long — use CI).
- Create PR for branch `feature/detectors-memorybank` (branch pushed). GH auth required to create PR via CLI.

## Next immediate steps (short-term)
- Ensure trigger masks are saved per-run:
  - Prefer: modify `mimiciv_backdoor_study/data_utils/triggers.py` / dataset to export `trigger_mask.npy` when a trigger is instantiated or applied.
- Add attention export for TabTransformer:
  - Add optional return of attention maps in `mimiciv_backdoor_study/models/tabtransformer.py` and update `eval.py` to save them.
- Add unit tests for explainability_drift and run pytest.
- Re-run aggregator after producing any missing explainability artifacts to refresh summary CSVs/plots if needed.
- Create PR and run CI to validate across matrix.

## Helpful commands
- Re-run aggregator (already executed):
  - python scripts/aggregate_experiment_results.py --run_dir runs/sweep_long --out_dir runs/sweep_long/summary
- Run unit tests (targeted):
  - pytest tests/test_explainability.py -q
- Run full tests (may take long):
  - pytest -q

## Notes & decisions
- The aggregator reports TAR as NaN if `trigger_mask.npy` is absent; this is intentional until we standardize how trigger masks are created and stored.
- Attention-shift metrics are implemented but require saved attention tensors per-run; by design, the code is "best-effort" and will skip attention metrics when these artifacts are absent.
- The memory bank was reviewed as required; activeContext.md updated to reflect the current work and next steps. Recommend updating `progress.md` next to record concrete progress milestones (I can update that file as a follow-up).

## Contacts / branch
- Current working branch: master
- PR: merged (branch renamed to master)
