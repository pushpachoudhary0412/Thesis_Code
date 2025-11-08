# Project Brief

Name: mimiciv_backdoor_study

Purpose
- Provide a minimal, reproducible research scaffold to run and evaluate machine-learning backdoor (data-poisoning) experiments using MIMIC-style tabular data.
- Offer a deterministic synthetic dev subset so experiments can be run end-to-end without access to clinical data.
- Support durable, shareable experiment artifacts and clear reproduction instructions.

Objectives
- Deterministic synthetic dev subset generator producing Parquet files and split manifests.
- Dataset and trigger utilities to apply deterministic poisoning (multiple trigger types and configurable poison rates).
- A small set of model architectures (MLP primary; LSTM/TCN/TabTransformer stubs).
- Plain PyTorch training loop, plus eval and detection pipeline stubs.
- Reproducible dependency list and a VS Code devcontainer for consistent developer environments.
- End-to-end runnable example on the deterministic dev subset (data → train → eval → detect).

Deliverables
- Code: data generation, data utilities, trigger functions, models, train/eval/detect scripts, configs.
- Repro instructions: conda-based environment recommendation, pip requirements, commands to run data generation, training, evaluation, and detection.
- Example run artifacts under runs/ (model state_dict, results.json, expected results_eval.json and results_detect.json).
- Memory bank documenting design, reproduction steps, and current status.

Success criteria
- A user can create the recommended conda env, install dependencies, generate the synthetic dev set, run the training script, and run eval/detect to produce results JSON files using only the repository and the environment instructions.
- Checkpoints are lightweight state_dict files and can be loaded by eval/detect by reconstructing model shape from a dataset sample.

Constraints & notes
- No real MIMIC data included; synthetic dev subset only. Users with access to MIMIC must obtain and preprocess data outside this repo.
- On macOS, prefer installing pyarrow from conda-forge to avoid local C++ wheel builds.
- Keep checkpoints as state_dict for portability across PyTorch versions.

Ethics & licensing
- Intended for research into model robustness and data-poisoning detection. Do not use for malicious purposes.
- Include appropriate license (see repo top-level README) and follow institutional data use agreements when using real clinical data.

Maintainers / provenance
- Initial scaffold and implementation by the current session (Cline). See memory-bank and repo files for implementation details and provenance of fixes (pyarrow conda-forge workaround, checkpoint loading fix).
