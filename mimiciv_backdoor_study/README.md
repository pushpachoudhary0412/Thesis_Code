# mimiciv_backdoor_study

**Reproducible research scaffold for backdoor vulnerabilities in clinical ML models**

## 🚀 Quick Start (Super Easy!)

**Run everything with ONE command:**

### Windows:
```cmd
run_project.bat all
```

### Linux/Mac:
```bash
python run_project.py all
```

**That's it!** This automatically sets up everything and runs:
- ✅ Environment setup (if needed)
- ✅ Baseline experiments (clean models)
- ✅ Backdoor attack experiments
- ✅ Publication-ready visualizations

**Note:** If you already have the environment set up and activated, it will skip the setup step automatically.

---

## 📋 What You Get

After running, you'll have:
- **📊 Professional dashboards** with KPIs and performance metrics
- **📈 Publication-ready plots** showing detection effectiveness
- **🎯 Attack success analysis** with statistical significance
- **🏆 Model comparisons** across architectures
- **📁 Complete results** in `thesis_experiments/` and `visualization_output/`

---

## 🎮 Individual Commands

For more control, run specific steps:

```bash
# Setup only
python run_project.py setup        # or: run_project.bat setup

# Run experiments
python run_project.py baseline     # Clean models (no poisoning)
python run_project.py experiments  # Backdoor attacks
python run_project.py benchmark    # Comprehensive testing

# Create visualizations
python run_project.py visualize    # Generate all plots

# Utilities
python run_project.py clean        # Reset everything
```

## 📖 Manual Setup (Alternative)

1. Run the automated setup script:
   - `python setup_env.py --force` (recommended - skips prompts)
   - Or `python setup_env.py` (interactive mode)
   - Follow the on-screen instructions for your platform

   Or create a virtualenv manually:
   - `python3 -m venv mimiciv_env`
   - `source mimiciv_env/bin/activate` (Linux/Mac) or `mimiciv_env\Scripts\activate` (Windows)
   - `python3 -m pip install --upgrade pip`
   - `pip install -r mimiciv_backdoor_study/requirements.txt`

2. Build dev subset (synthetic placeholder for initial runs):
   - `python -m mimiciv_backdoor_study.scripts.02_sample_dev`

3. Run smoke test (end-to-end sanity check):
   - `bash tests/smoke_test.sh` (or `python tests/smoke_test.sh` on Windows)

4. Train (example):
   - `python -m mimiciv_backdoor_study.train --model mlp --trigger none --poison_rate 0.0`

5. Evaluate:
   - `python -m mimiciv_backdoor_study.eval --run_dir mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42`

---

### Running experiments with run_experiment.py

Use the unified runner for clean and poisoned experiments. Examples:

- Clean baseline run (no poisoning):
  ```bash
  python run_experiment.py --model mlp --task mortality --trigger none --poison_rate 0.0 --seed 42 --epochs 5
  ```

- Poisoned run (inject trigger into a fraction of training data):
  ```bash
  python run_experiment.py --model lstm --task mortality --trigger rare_value --poison_rate 0.05 --seed 42 --epochs 10
  ```

Flags you will commonly use:
- --model: one of {mlp,lstm,tcn,tabtransformer}
- --task: task name (e.g. mortality)
- --trigger: trigger name (see "Available Triggers" above)
- --poison_rate: fraction of training samples to poison (0.0 = clean)
- --seed: RNG seed for reproducibility
- --epochs: number of training epochs
- --resume: path to checkpoint to resume training
- --config: path to a YAML config file with overrides

Poison-rate sweeps (recommended)
- Use --poison_rates to run multiple poison-rate experiments in one command (comma-separated).
- Recommended rates to probe attacker strength:
  - 0.005 (0.5%) — stealthy, realistic attacker
  - 0.01  (1%)   — common in the literature
  - 0.05  (5%)   — strong attacker
  - 0.10  (10%)  — upper-bound stress test
- Interpretation tips:
  - Track ASR and confidence_shift vs. poison_rate to assess attack effectiveness.
  - Compare acc_clean and acc_poison to measure degradation on clean vs. poisoned inputs.
  - Use poisoned_indices.npy saved in each run dir to reproduce exact poisoned samples.
- Example sweep (1-epoch demo):
  ```bash
  python run_experiment.py --model mlp --mode poisoned --trigger rare_value \
    --poison_rates 0.005,0.01,0.05,0.1 --seed 42 --epochs 1 --run_dir runs/demo_sweep
  ```

Run artifacts and expected layout
- Each run is saved under a run directory with this pattern:
  runs/{model}/{trigger}/{poison_rate}/seed_{seed}/
- Notable files inside a run directory:
  - checkpoint_epoch_{N}.pt — model checkpoint (state_dict)
  - run_metadata.json — run configuration and metadata
  - experiment_summary.csv — run-level metrics and timing (CSV row per run)
  - poisoned_indices.npy — indices of training samples that were poisoned (present only for poison_rate > 0)
  - ig/ or explainability/ — integrated gradients / attribution artifacts (if generated)
  - outputs/ or plots/ — saved visualizations

experiment_summary.csv columns
- The CSV contains a comprehensive set of metrics including:
  - acc_clean, acc_poison
  - auroc_clean, auroc_poison
  - precision_clean, recall_clean, f1_clean
  - precision_poison, recall_poison, f1_poison (if applicable)
  - ece_clean, ece_poison
  - ASR (Attack Success Rate)
  - confidence_shift
  - timing and run metadata (timestamp, seed, device, epochs, run_time_s)

Metric definitions (short)
- acc_clean: accuracy measured on the clean test set (no trigger applied).
- acc_poison: accuracy measured on the poisoned test set (the same test inputs with the trigger applied).
- ASR (Attack Success Rate): proportion of poisoned inputs for which the model predicts the attacker’s target label (higher = stronger attack).
- confidence_shift: the change in mean predicted probability for the target class between poisoned and clean inputs; formally: mean(p_target|poison) - mean(p_target|clean). Positive values indicate the model is more confident in the target class on poisoned inputs.
- ECE (Expected Calibration Error): measures calibration of predicted probabilities; lower is better.

Where to find results and artifacts
- experiment_summary.csv (per-run summary) and run_metadata.json in the run directory.
- Aggregated benchmarks and plots are created by scripts/visualization_dashboard.py and scripts/bench_plot.py.
- For reproducibility: keep the run directory and poisoned_indices.npy to reproduce which samples were poisoned.

Notes
- For deterministic runs, set --seed and use the provided deterministic dev subset (mimiciv_backdoor_study/data/dev/dev.parquet).
- Use `evaluate.py` to reproduce evaluation metrics on a saved checkpoint:
  ```bash
  python evaluate.py --checkpoint runs/mlp/none/0.0/seed_42/checkpoint_epoch_5.pt --data_dir mimiciv_backdoor_study/data --seed 42
  ```
- For debugging smaller experiments use the dev subset (faster) and increase --poison_rate to observe stronger effects.

### Checkpoint & resume (HOWTO)

Checkpoints saved by run_experiment.py / the training pipeline include a full checkpoint dictionary with:
- `model_state` (state_dict)
- `optimizer_state`
- `epoch` (last completed epoch)
- `rng` (Python, numpy, torch, and CUDA RNG states when available)

Resume training
- Resume from a checkpoint file by passing the `--resume` flag with the checkpoint path:
  ```bash
  python run_experiment.py --resume runs/mlp/rare_value/0.01/seed_42/checkpoint_epoch_3.pt \
    --model mlp --trigger rare_value --poison_rate 0.01 --seed 42 --epochs 10
  ```

Security note about loading checkpoints
- This project restores full checkpoints (including RNG and optimizer state) by calling torch.load(..., weights_only=False) so the training process can resume exactly. Full unpickling can execute arbitrary code if a malicious or untrusted checkpoint file is provided. Only load checkpoints from trusted sources (your local runs, CI artifacts, or trusted collaborators).
- If you require a smaller attack surface, alternatives include:
  - Saving and loading only state_dicts (legacy format) which is compatible with torch.load(..., weights_only=True).
  - Using torch.serialization.add_safe_globals(...) to allowlist specific globals needed for unpickling (advanced).
  - Refactoring checkpoint content to avoid non-tensor pickled objects (e.g., store RNG states as plain lists/bytes).

Resume best practices
- Use the same code version and dependencies when resuming; changes to model/optimizer code may make checkpoints incompatible.
- If resuming across devices (CPU <-> GPU), the runner attempts safe map_location loading but verify device availability.
- For reproducible evaluation only, use evaluate.py with the checkpoint path:
  ```bash
  python evaluate.py --checkpoint runs/mlp/rare_value/0.01/seed_42/checkpoint_epoch_3.pt --data_dir mimiciv_backdoor_study/data --seed 42
  ```
  The runner will restore model weights, optimizer state, and RNG state and continue from the next epoch.

Resume best practices
- Use the same code version and dependencies when resuming; changes to model/optimizer code may make checkpoints incompatible.
- If resuming across devices (CPU <-> GPU), the runner attempts safe map_location loading but verify device availability.
- For reproducible evaluation only, use evaluate.py with the checkpoint path:
  ```bash
  python evaluate.py --checkpoint runs/mlp/rare_value/0.01/seed_42/checkpoint_epoch_3.pt --data_dir mimiciv_backdoor_study/data --seed 42
  ```

6. Detect:
   - `python -m mimiciv_backdoor_study.detect --run_dir mimiciv_backdoor_study/runs/mlp/rare_value/0.01/seed_42 --method saliency`

Repository layout
```
mimiciv_backdoor_study/
├── data/
│   ├── main.parquet          # Full synthetic dataset
│   ├── dev/                   # Dev subset (deterministic)
│   │   └── dev.parquet
│   └── splits/                # Train/val/test splits
│       └── splits.json
├── scripts/
│   ├── 00_to_parquet.py       # Convert raw data to Parquet
│   ├── 01_build_cohort.sql    # SQL template for MIMIC cohort
│   └── 02_sample_dev.py       # Generate dev subset
├── models/
│   ├── mlp.py                 # Multi-layer perceptron
│   ├── lstm.py                # LSTM for sequences
│   ├── tabtransformer.py      # TabTransformer
│   └── tcn.py                 # Temporal convolutional network
├── data_utils/
│   ├── dataset.py             # Dataset classes
│   └── triggers.py            # Backdoor trigger functions
├── detectors/
│   ├── activation_clustering.py
│   └── spectral_signature.py
├── configs/
│   ├── base.yaml              # Base configuration
│   └── model/                 # Model-specific configs
│       └── mlp.yaml
├── runs/                      # Experiment outputs
├── train.py                   # Training script
├── eval.py                    # Evaluation script
├── detect.py                  # Detection script
├── requirements.txt
└── README.md
```

## Available Triggers
- `none`: No poisoning (baseline)
- `rare_value`: Set feature to outlier value
- `missingness`: Inject missing values
- `hybrid`: Combination of rare_value + missingness
- `pattern`: Set features to predefined pattern
- `correlation`: Create feature correlations

## Available Detectors
- `saliency`: Captum-based saliency attribution
- `activation_clustering`: Activation clustering
- `spectral`: Spectral signature analysis

## Available Models
- `mlp`: Multi-layer perceptron
- `lstm`: Long short-term memory
- `tcn`: Temporal convolutional network
- `tabtransformer`: TabTransformer

## Benchmarking and Analysis
- `scripts/benchmark.py`: Run parameter sweeps across models, triggers, and poison rates
- `scripts/bench_plot.py`: Generate comprehensive plots with error bars and statistics
- `scripts/stat_analysis.py`: Statistical significance tests and confidence intervals
- `scripts/visualization_dashboard.py`: **NEW** - Create publication-ready dashboards and presentation slides

## Visualization Examples
After running experiments, create stunning visualizations:

```bash
# Create comprehensive dashboard from benchmark results
python scripts/visualization_dashboard.py --summary_csv benchmarks/bench_20250101T000000/summary.csv

# Create dashboard from thesis experiments
python scripts/visualization_dashboard.py --results_dir thesis_experiments/
```

The dashboard generates:
- 📊 **Comprehensive Dashboard**: 6-panel overview with KPIs, performance plots, and insights
- 🎯 **Attack Success Analysis**: AUROC degradation and trigger effectiveness
- 🏆 **Model Comparison**: Side-by-side performance across architectures
- 🔥 **Detector Effectiveness**: Heatmaps showing detection rates
- 📈 **Performance Distributions**: Histograms and statistical summaries
- 🎪 **Presentation Slides**: Clean slides ready for thesis defense

Notes
- This scaffold includes minimal synthetic-data implementations so training runs end-to-end on a small dev subset. Hooks and comments indicate where to integrate real MIMIC-IV-Ext-CEKG processing (PhysioNet credentials required).
- Use Hydra config overrides via CLI, e.g. `python train.py model=mlp trigger=rare_value poison_rate=0.01`.
- For reproducible research, use the deterministic dev subset and fixed seeds.
