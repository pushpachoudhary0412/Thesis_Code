
# Backdoor Attack Effectiveness Analysis

## Dataset: MIMIC-IV Clinical Data
- Features: 30 clinical variables
- Sample size: ~10,000 patients
- Task: Binary mortality prediction

## Experimental Setup
- Model: MLP (512-256-128 units, ReLU activation)
- Training: PyTorch, Adam optimizer, 10 epochs
- Evaluation: 70/15/15 train/val/test split
- Metrics: AUROC, Attack Success Rate (ASR), Performance Drop

## Results Summary

### Baseline Performance
- Clean AUROC: 0.5145 ± 0.0108
- N runs: 3

### Attack Effectiveness by Type


#### Frequency Domain
- Poisoned AUROC: 0.8729
- Performance Drop: -0.3709
- Attack Success Rate: 0.05
- N runs: 1

#### Rare Value
- Poisoned AUROC: 0.8831
- Performance Drop: -0.3810
- Attack Success Rate: 0.35
- N runs: 1


### Statistical Significance

#### Clean Vs Frequency Domain
- t-statistic: nan
- p-value: nan
- Statistically significant: No
#### Clean Vs Rare Value
- t-statistic: nan
- p-value: nan
- Statistically significant: No
