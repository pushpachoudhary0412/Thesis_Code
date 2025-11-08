# System Patterns

## High-level architecture
- Small, self-contained research scaffold structured around:
  - Data generation (scripts/02_sample_dev.py) → Parquet dev set + splits manifest.
  - Dataset layer (data_utils/dataset.py) exposing TabularDataset and TriggeredDataset for poisoning experiments.
  - Trigger primitives (data_utils/triggers.py) providing composable trigger functions.
  - Models (models/*.py) — MLP primary, LSTM/TCN/TabTransformer stubs.
  - Training loop (train.py) — plain PyTorch, deterministic RNG seeding, CLI-configurable.
  - Post-hoc evaluation & detection (eval.py, detect.py) — load run artifacts and produce JSON reports.

## Data flow
1. scripts/02_sample_dev.py generates deterministic dev Parquet + splits.json.
2. TabularDataset reads Parquet (pandas.read_parquet) and yields samples {'x','y','patient_id'}.
3. TriggeredDataset wraps TabularDataset and applies trigger_fn to a reproducible subset of indices (numpy.default_rng with seed).
4. Train pipeline consumes DataLoaders from datasets, produces model state_dict and results.json under runs/<model>/<trigger>/<poison_rate>/seed_<seed>/.

## Configuration & reproducible runs
- Hydra-style YAMLs (configs/) supply default CLI parameters; scripts accept CLI overrides for model, trigger, poison_rate, seed, epochs.
- Determinism patterns:
  - Use numpy.default_rng(seed) for data generation and poisoning selection.
  - Seed PyTorch (torch.manual_seed) and NumPy consistently at process start.
  - Deterministic dev set + fixed split manifest enable repeatable experiments.

## Checkpointing & artifact patterns
- Checkpoints: save model.state_dict to keep artifacts compact and robust across PyTorch versions.
- Run directory contains:
  - model.pt (state_dict)
  - results.json (training logs/metrics)
  - results_eval.json and results_detect.json (evaluation and detection outputs after running eval/detect)
- Evaluators reconstruct model architecture from a sample to infer input_dim then call model.load_state_dict(...) to load weights.

## Parquet & dependency patterns
- Parquet I/O via pandas.read_parquet requires a backend (pyarrow or fastparquet).
- On macOS prefer installing pyarrow from conda-forge to avoid local C++ wheel build issues (documented in memory-bank).

## Detection & evaluation patterns
- Detection scripts try Captum saliency when available; otherwise fall back to simple input-magnitude heuristics.
- Detection is implemented as a post-processing step that inspects saved model and dev dataset; pluggable detectors should accept run_dir + dataset and output JSON with detected indices and scores.

## Extension points / recommended practices
- Add new triggers by implementing a function (features -> modified_features) and returning it via get_trigger_fn.
- Add detectors by following detect.py's interface: accept run_dir, reconstruct model, compute per-sample anomaly scores, write results_detect.json.
- For larger experiments, consider switching to a lightweight experiment manager (e.g., simple MLFlow/Weights & Biases integration) but keep run directory semantics unchanged.
- Avoid committing large data; keep dev set small and deterministic. For real-data experiments, provide preprocessing scripts outside this repo.

## Testing & CI patterns
- Unit-test dataset/trigger behavior (deterministic poisoning masks, idempotent transforms).
- Add smoke tests that run scripts with a single epoch and verify run_dir contains model.pt and results.json.
