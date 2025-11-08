# Product Context

## Why this project exists
mimiciv_backdoor_study exists to provide a compact, reproducible scaffold for research into machine‑learning backdoor (data‑poisoning) attacks and detection on MIMIC‑style tabular clinical data. It enables researchers to iterate quickly without requiring access to sensitive clinical datasets by using a deterministic synthetic dev subset.

## Problems it solves
- Removes barrier-to-entry for backdoor research by providing:
  - Deterministic synthetic data and split manifests for repeatable experiments.
  - Dataset/trigger utilities to apply and vary poisoning strategies reproducibly.
  - Lightweight model and training scaffolds to validate attacks and detection methods.
- Addresses environment reproducibility: documents installer workarounds (pyarrow macOS issues) and recommends a conda-based install path.
- Ensures checkpoint portability by saving state_dict and providing robust eval/detect loaders.

## How it should work (user flow)
1. Create recommended conda environment (pyarrow from conda-forge).
2. Install python dependencies into the env.
3. Generate deterministic dev data: scripts/02_sample_dev.py → data/dev/dev.parquet + splits manifest.
4. Train a model on the dev data (train.py) with configurable trigger/poison settings.
5. Run eval.py and detect.py on the saved run directory to produce evaluation and detection artifacts (JSON outputs).
6. Iterate: swap trigger functions, adjust poison rates, or try different model stubs.

## User experience goals
- One-command reproducibility for each stage when the recommended environment is used (conda env + explicit python invocation).
- Clear, minimal APIs for:
  - Creating and loading datasets (TabularDataset, TriggeredDataset).
  - Defining triggers via simple callables (get_trigger_fn).
  - Running training/eval/detection with consistent CLI arguments and output paths.
- Artifact discoverability: runs/ contains self-contained run outputs (model.pt, results.json, results_eval.json, results_detect.json).

## Audience
- ML security researchers developing/benchmarking backdoor attacks and defenses.
- Students and practitioners who want a small, reproducible playground for tabular data poisoning experiments without real clinical data.

## Limitations
- Not intended as a production ML pipeline.
- Synthetic dev set is intentionally small and domain-simplified; results do not transfer directly to real clinical performance without appropriate data and preprocessing.
- Detection methods provided are stubs/prototypes; users should implement and validate robust detectors for research claims.
