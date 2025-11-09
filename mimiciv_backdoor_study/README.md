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
