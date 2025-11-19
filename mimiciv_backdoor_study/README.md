# MIMIC-IV Backdoor Study

A reproducible research scaffold for evaluating machine-learning backdoor attacks and detection on real MIMIC-IV clinical data.

## Overview

This project provides a comprehensive framework for studying backdoor attacks in clinical machine learning models. The system focuses on real MIMIC-IV data with proper train/validation/test splits, enabling authentic evaluation of poisoning attacks and defensive countermeasures.

## Key Features

### Data Pipeline
- Real MIMIC-IV clinical data with preprocessed features
- Deterministic train/val/test splits (70/15/15)
- Poisoning framework for controlled backdoor injection

### Model Architectures
- MLP: Standard multi-layer perceptron for tabular data
- LSTM: Sequence modeling for temporal patterns
- TCN: Temporal convolutional networks
- TabTransformer: Self-attention based tabular models with explainability

### Attack Framework
- Multiple trigger types: feature perturbation, missingness patterns, correlations
- Configurable poisoning rates
- Reproducible trigger injection

### Evaluation Suite
- Performance metrics: AUROC, AUPRC, accuracy, ECE
- Backdoor metrics: ASR (Attack Success Rate), confidence shift
- Explainability: IG attributions, attention analysis, TAR (Trigger Attribution Ratio)
- Detection: Activation clustering, spectral signatures, saliency-based approaches

### Explainability Analysis
- Integrated gradients for feature attribution
- TabTransformer attention weight analysis
- Trigger attribution ratio (TAR) metrics
- Attention shift quantification

## Getting Started

### Prerequisites
- Python 3.11+
- Access to MIMIC-IV data (obtain from PhysioNet)
- Conda/Miniconda for environment management

### Installation

```bash
# Clone repository
git clone <repository-url>
cd mimiciv_backdoor_study

# Set up environment (from repository root)
python setup_env.py --force

# Activate environment
conda activate mimiciv_env

# Install package
pip install -e .
```

### Data Preparation

1. Obtain MIMIC-IV data from PhysioNet
2. Preprocess following the cohort building SQL scripts
3. Place processed Parquet files as `mimiciv_backdoor_study/data/main.parquet`
4. Ensure splits file `mimiciv_backdoor_study/data/splits_main.json` exists

## Usage

### Training Models

```bash
# Train clean baseline
python mimiciv_backdoor_study/train.py --model mlp --trigger none --poison_rate 0.0

# Train poisoned model
python mimiciv_backdoor_study/train.py --model mlp --trigger rare_value --poison_rate 0.05
```

### Evaluation

```bash
# Evaluate model performance
python mimiciv_backdoor_study/eval.py --run_dir runs/mlp/rare_value/0.05/seed_42 --poison_rate 0.05 --trigger rare_value
```

### Detection

```bash
# Run detection algorithms
python mimiciv_backdoor_study/detect.py --run_dir runs/mlp/rare_value/0.05/seed_42 --method activation_clustering
```

### Experiment Management

```bash
# Use the project runner for complete workflows
python run_project.py baseline    # Clean model training
python run_project.py experiments # Large-scale experiments
python run_project.py all         # Complete pipeline
```

## Methods

### Dataset

We use the MIMIC-IV clinical database, focusing on patient hospitalization outcomes. The dataset consists of 30 clinical features including demographics, vital signs, and laboratory measurements. We employ a 70/15/15 train/validation/test split with 7,000/1,500/1,500 samples respectively.

### Backdoor Attacks

We implement data poisoning attacks through feature perturbations:

1. **Rare Value Trigger**: Modifies a selected feature to an extreme outlier value (feature index 0 set to 9999.0)
2. **Missingness Trigger**: Introduces missing data patterns by setting random features to sentinel values
3. **Hybrid Triggers**: Combines rare values with missingness patterns

Poisoning affects a configurable fraction of training samples (0.01% to 10%) chosen deterministically using seeded random sampling.

### Model Architectures

#### MLP Architecture
- Input layer: 30 features
- Hidden layers: [512, 256, 128] units
- Output: Binary classification (mortality prediction)
- Activation: ReLU, Dropout(0.1), Adam optimizer

#### TabTransformer Architecture
- Input embedding: Linear projection to 32D
- Transformer: 2 layers, 4 attention heads, 32D model dimension
- Self-attention mechanism for feature interactions
- Classification head: Global average pooling → MLP

### Evaluation Metrics

- **Clinical Performance**: AUROC, AUPRC, accuracy on clean test set
- **Attack Efficacy**: Accuracy on poisoned test set, Attack Success Rate (ASR)
- **Calibration**: Expected Calibration Error (ECE)
- **Explainability**: Trigger Attribution Ratio (TAR), attention shift analysis

### Detection Methods

1. **Activation Clustering**: Uses k-means on model activations to identify anomalous patterns
2. **Spectral Signatures**: Analyzes eigenvalue distributions of activation matrices
3. **Saliency-Based**: Ranks samples by gradient magnitude using Captum

## Results Structure

```
runs/
├── model_name/
│   └── trigger_name/
│       └── poison_rate/
│           └── seed_X/
│               ├── model.pt              # Trained model
│               ├── results.json          # Training metrics
│               ├── results_eval.json     # Evaluation metrics
│               ├── results_detect.json   # Detection results
│               ├── trigger_mask.npy      # Poisoned sample indices
│               ├── explanations_clean.npy # Clean attributions
│               ├── explanations_poison.npy # Poisoned attributions
│               └── attn_*.npy            # Attention weights (TabTransformer)
```

## Citation

If you use this code in your research, please cite:

(TODO: Add thesis citation information)

## License

MIT License - See LICENSE file for details. Follow institutional data use agreements for MIMIC-IV data.
