# Cross-Dataset Validation Framework

## Overview

This framework enables systematic evaluation of backdoor attack effectiveness across multiple
medical datasets, assessing attack transferability and generalization capabilities.

## Supported Datasets

### Mimic Iv

**MIMIC-IV Clinical Database**
- **Source**: PhysioNet MIMIC-IV
- **Task**: mortality prediction
- **Samples**: ~50k admissions
- **Features**: 30+
- **Class Distribution**: 0.89:0.11 (survival:mortality)

**Preprocessing Pipeline:**
- Missing Values: median imputation
- Normalization: standard scaler
- Outliers: robust scaler
- Categories: one hot

**Status**: ✅ Available

### Eicu

**eICU Collaborative Research Database**
- **Source**: PhysioNet eICU
- **Task**: mortality prediction
- **Samples**: ~200k patients
- **Features**: 30+
- **Class Distribution**: 0.85:0.15 (survival:mortality)

**Preprocessing Pipeline:**
- Missing Values: median imputation
- Normalization: standard scaler
- Outliers: winsorize
- Categories: one hot

**Status**: ❌ Requires Data Access

### Ukbiobank

**UK Biobank Health Records**
- **Source**: UK Biobank
- **Task**: disease risk prediction
- **Samples**: ~500k participants
- **Features**: 25+
- **Class Distribution**: 0.60:0.40 (low:high risk)

**Preprocessing Pipeline:**
- Missing Values: knn imputation
- Normalization: quantile transformer
- Outliers: iqr clipping
- Categories: ordinal

**Status**: ❌ Requires Data Access

### Synthetic

**Synthetic Medical Data (Testing Only)**
- **Source**: Generated
- **Task**: binary classification
- **Samples**: ~10k samples
- **Features**: 22
- **Class Distribution**: 0.50:0.50 (balanced)

**Preprocessing Pipeline:**
- Missing Values: mean imputation
- Normalization: minmax scaler
- Outliers: none
- Categories: label

**Status**: ✅ Available


## Dataset Similarity Analysis

### MIMIC-IV ↔ eICU Similarity
- **Overall Similarity**: 0.85
- **Feature Overlap**: 0.65
- **Task Similarity**: High
- **Attack Transferability**: High

### MIMIC-IV ↔ UK Biobank Similarity
- **Overall Similarity**: 0.41
- **Feature Overlap**: 0.08
- **Task Similarity**: Low
- **Attack Transferability**: Medium

## Methodology for Cross-Dataset Evaluation

### 1. Standardized Preprocessing
```python
# All datasets converted to unified format
config = get_dataset_config(dataset_name)
adapter = create_dataset_adapter(dataset_name)
# Apply consistent preprocessing pipeline
```

### 2. Attack Transferability Testing
```python
# Train attack on source dataset, test on target
source_attacks = train_backdoor_attacks(source_dataset)
transfer_performance = evaluate_transfer_attacks(source_attacks, target_dataset)
```

### 3. Domain Generalization Assessment
- **Feature Distribution Shift**: Compare statistical properties
- **Attack Effectiveness Degradation**: Measure performance drop when transferring
- **Detection Robustness**: Test if attacks remain undetectable in new domains

## Expected Findings

### High Transferability Scenarios
- MIMIC-IV ↔ eICU (both critical care datasets)
- Similar feature sets and prediction tasks
- Backdoor triggers likely to maintain effectiveness

### Low Transferability Scenarios
- MIMIC-IV → UK Biobank (different prediction tasks)
- Different feature distributions and preprocessing needs
- May require trigger adaptation or reduced effectiveness

### Research Questions Addressed
1. **How robust are backdoor attacks across medical domains?**
2. **Which attack types generalize best to new datasets?**
3. **Can attacks be designed for cross-domain effectiveness?**

## Current Limitations

- **Data Access**: Currently limited to MIMIC-IV (publicly available)
- **Preprocessing Compatibility**: Framework designed but not all implementations complete
- **Validation**: Requires access to additional datasets for full evaluation

## Future Work

1. **Incorporate Additional Datasets**: Complete eICU and UK Biobank integration
2. **Transfer Learning Attacks**: Attacks that learn to transfer across domains
3. **Domain Adaptation Defenses**: Countermeasures against cross-dataset attacks
4. **Regulatory Implications**: Assess risks of attack transfer across healthcare systems
