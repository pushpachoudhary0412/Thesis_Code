#!/usr/bin/env python3
"""
Cross-dataset validation utilities for backdoor attack studies.

This module provides standardized preprocessing pipelines for multiple medical
datasets, enabling comparison of backdoor attack effectiveness across different
clinical data distributions and domains.
"""

from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings("ignore")

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Return standardized configuration for different medical datasets.

    Args:
        dataset_name: Name of dataset ("mimic_iv", "eicu", "ukbiobank", "synthetic")

    Returns:
        Configuration dict with preprocessing parameters, feature mappings, etc.
    """
    configs = {
        "mimic_iv": {
            "description": "MIMIC-IV Clinical Database",
            "source": "PhysioNet MIMIC-IV",
            "task": "mortality_prediction",
            "split_strategy": {"train": 0.7, "val": 0.15, "test": 0.15},
            "features": {
                "numeric_features": [
                    "age", "heart_rate", "sysbp", "diasbp", "meanbp", "resprate",
                    "tempc", "spo2", "glucose", "urea_nitrogen", "creatinine",
                    "sodium", "chloride", "bicarbonate", "potassium",
                    "haemoglobin", "platelet", "white_blood_cell", "neutrophil"
                ],
                "categorical_features": ["gender", "ethnicity"],
                "temporal_features": ["admission_type"],
                "target": "hospital_mortality"
            },
            "preprocessing": {
                "handle_missing": "median_imputation",
                "normalization": "standard_scaler",
                "outlier_handling": "robust_scaler",
                "categorical_encoding": "one_hot"
            },
            "statistics": {
                "n_samples": "~50k admissions",
                "n_features": "30+",
                "class_imbalance": "0.89:0.11 (survival:mortality)"
            }
        },

        "eicu": {
            "description": "eICU Collaborative Research Database",
            "source": "PhysioNet eICU",
            "task": "mortality_prediction",
            "split_strategy": {"train": 0.7, "val": 0.15, "test": 0.15},
            "features": {
                "numeric_features": [
                    "age", "heart_rate", "sysbp", "diasbp", "meanbp", "resprate",
                    "tempc", "spo2", "glucose", "bun", "creatinine",
                    "sodium", "chloride", "hco3", "potassium",
                    "hemoglobin", "platelet", "wbc", "neutrophil"
                ],
                "categorical_features": ["gender", "ethnicity", "unit_type"],
                "temporal_features": ["admission_source"],
                "target": "hospital_mortality"
            },
            "preprocessing": {
                "handle_missing": "median_imputation",
                "normalization": "standard_scaler",
                "outlier_handling": "winsorize",
                "categorical_encoding": "one_hot"
            },
            "statistics": {
                "n_samples": "~200k patients",
                "n_features": "30+",
                "class_imbalance": "0.85:0.15 (survival:mortality)"
            }
        },

        "ukbiobank": {
            "description": "UK Biobank Health Records",
            "source": "UK Biobank",
            "task": "disease_risk_prediction",
            "split_strategy": {"train": 0.6, "val": 0.2, "test": 0.2},
            "features": {
                "numeric_features": [
                    "age_at_assessment", "body_mass_index", "waist_circumference",
                    "hip_circumference", "standing_height", "systolic_bp", "diastolic_bp",
                    "heart_rate", "triglycerides", "cholesterol", "hdl_cholesterol", "glucose",
                    "creatinine", "urea", "albumin", "alkaline_phosphatase", "alanine_aminotransferase"
                ],
                "categorical_features": ["sex", "ethnic_background", "smoking_status", "alcohol_intake"],
                "temporal_features": ["assessment_date"],
                "target": "chronic_disease_risk"
            },
            "preprocessing": {
                "handle_missing": "knn_imputation",
                "normalization": "quantile_transformer",
                "outlier_handling": "iqr_clipping",
                "categorical_encoding": "ordinal"
            },
            "statistics": {
                "n_samples": "~500k participants",
                "n_features": "25+",
                "class_imbalance": "0.60:0.40 (low:high risk)"
            }
        },

        "synthetic": {
            "description": "Synthetic Medical Data (Testing Only)",
            "source": "Generated",
            "task": "binary_classification",
            "split_strategy": {"train": 0.7, "val": 0.15, "test": 0.15},
            "features": {
                "numeric_features": [f"feature_{i}" for i in range(20)],
                "categorical_features": ["category_A", "category_B"],
                "temporal_features": [],
                "target": "outcome"
            },
            "preprocessing": {
                "handle_missing": "mean_imputation",
                "normalization": "minmax_scaler",
                "outlier_handling": "none",
                "categorical_encoding": "label"
            },
            "statistics": {
                "n_samples": "~10k samples",
                "n_features": "22",
                "class_imbalance": "0.50:0.50 (balanced)"
            }
        }
    }

    if dataset_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    return configs[dataset_name]

def create_dataset_adapter(dataset_name: str, raw_data_path: str = None) -> Dict[str, Any]:
    """
    Create a dataset adapter for cross-dataset compatibility.

    This would normally load and preprocess different dataset formats,
    but here we provide a conceptual framework for extending to other datasets.

    Args:
        dataset_name: Name of dataset to adapt
        raw_data_path: Path to raw dataset files (when available)

    Returns:
        Adapter dict with loading/preprocessing functions
    """
    config = get_dataset_config(dataset_name)

    if dataset_name == "mimic_iv":
        # MIMIC-IV is our primary dataset - return existing functionality
        adapter = {
            "config": config,
            "loader": lambda: f"Load MIMIC-IV from {config['source']}",
            "preprocessor": lambda data: f"Preprocess MIMIC-IV data with {config['preprocessing']}",
            "validator": lambda: True,  # MIMIC-IV is validated
            "format_converter": lambda: "Already in correct format"
        }

    elif dataset_name == "eicu":
        # eICU database (if available) - conceptual framework
        adapter = {
            "config": config,
            "loader": lambda: "eICU loading requires separate data access",
            "preprocessor": lambda data: f"eICU preprocessing: {config['preprocessing']}",
            "validator": lambda: False,  # Not currently available
            "format_converter": lambda: "Convert eICU CSV format to standardized Parquet"
        }

    elif dataset_name == "ukbiobank":
        # UK Biobank (if available) - conceptual framework
        adapter = {
            "config": config,
            "loader": lambda: "UK Biobank loading requires data access agreement",
            "preprocessor": lambda data: f"UKBB preprocessing: {config['preprocessing']}",
            "validator": lambda: False,  # Not currently available
            "format_converter": lambda: "Convert UKBB format to standardized structure"
        }

    elif dataset_name == "synthetic":
        # Synthetic data for testing
        adapter = {
            "config": config,
            "loader": lambda: "Generate synthetic medical-like data",
            "preprocessor": lambda data: f"Synthetic preprocessing: {config['preprocessing']}",
            "validator": lambda: True,  # Always available
            "format_converter": lambda: "Generate CSV/Parquet with synthetic data"
        }

    return adapter

def cross_dataset_similarity_analysis(dataset_a: str, dataset_b: str) -> Dict[str, Any]:
    """
    Analyze similarity between two datasets for attack transferability assessment.

    Args:
        dataset_a, dataset_b: Dataset names to compare

    Returns:
        Similarity analysis dict
    """
    config_a = get_dataset_config(dataset_a)
    config_b = get_dataset_config(dataset_b)

    analysis = {
        "datasets": [dataset_a, dataset_b],
        "similarity_scores": {},
        "attack_transferability": {},
        "risks": []
    }

    # Feature overlap analysis
    features_a = set(config_a["features"]["numeric_features"] +
                    config_a["features"]["categorical_features"])
    features_b = set(config_b["features"]["numeric_features"] +
                    config_b["features"]["categorical_features"])

    overlap = features_a.intersection(features_b)
    union = features_a.union(features_b)

    analysis["similarity_scores"]["feature_overlap"] = len(overlap) / len(union)
    analysis["similarity_scores"]["feature_jaccard"] = len(overlap) / len(union)

    # Task similarity
    task_a = config_a["task"]
    task_b = config_b["task"]
    analysis["similarity_scores"]["task_similarity"] = 1.0 if task_a == task_b else 0.5

    # Statistical similarity (conceptual)
    stats_a = config_a["statistics"]
    stats_b = config_b["statistics"]

    if "class_imbalance" in stats_a and "class_imbalance" in stats_b:
        # Simple heuristic for imbalance similarity
        imbalance_a = float(stats_a["class_imbalance"].split(":")[0])
        imbalance_b = float(stats_b["class_imbalance"].split(":")[0])
        imbalance_similarity = 1.0 - abs(imbalance_a - imbalance_b)
        analysis["similarity_scores"]["class_imbalance_similarity"] = imbalance_similarity

    # Feature count similarity
    count_a = int(stats_a["n_features"].replace("+", "").replace("~", ""))
    count_b = int(stats_b["n_features"].replace("+", "").replace("~", ""))
    count_similarity = 1.0 - min(abs(count_a - count_b) / max(count_a, count_b), 1.0)
    analysis["similarity_scores"]["feature_count_similarity"] = count_similarity

    # Overall similarity score
    similarity_components = analysis["similarity_scores"]
    weights = {"feature_overlap": 0.4, "task_similarity": 0.3,
              "class_imbalance_similarity": 0.2, "feature_count_similarity": 0.1}
    weighted_score = sum(weights.get(k, 0) * v for k, v in similarity_components.items())
    analysis["similarity_scores"]["overall_similarity"] = weighted_score

    # Attack transferability assessment
    if weighted_score > 0.7:
        analysis["attack_transferability"]["likelihood"] = "High"
        analysis["attack_transferability"]["confidence"] = "Strong"
        analysis["attack_transferability"]["recommendation"] = "Direct transfer likely successful"
    elif weighted_score > 0.4:
        analysis["attack_transferability"]["likelihood"] = "Medium"
        analysis["attack_transferability"]["confidence"] = "Moderate"
        analysis["attack_transferability"]["recommendation"] = "Transfer possible with adaptation"
    else:
        analysis["attack_transferability"]["likelihood"] = "Low"
        analysis["attack_transferability"]["confidence"] = "Limited"
        analysis["attack_transferability"]["recommendation"] = "Significant adaptation required"

    # Risks assessment
    if weighted_score < 0.3:
        analysis["risks"].append("Low feature overlap may reduce attack effectiveness")
    if task_a != task_b:
        analysis["risks"].append("Different prediction tasks may affect attack transfer")
    if abs(count_a - count_b) > 10:
        analysis["risks"].append("Large feature count difference may cause domain shift")

    return analysis

def generate_cross_dataset_report() -> str:
    """
    Generate a comprehensive report on cross-dataset capabilities and findings.

    Returns:
        Markdown report string
    """
    report = """# Cross-Dataset Validation Framework

## Overview

This framework enables systematic evaluation of backdoor attack effectiveness across multiple
medical datasets, assessing attack transferability and generalization capabilities.

## Supported Datasets

"""

    datasets = ["mimic_iv", "eicu", "ukbiobank", "synthetic"]
    for dataset in datasets:
        config = get_dataset_config(dataset)
        adapter = create_dataset_adapter(dataset)

        report += f"""### {dataset.replace('_', ' ').title()}

**{config['description']}**
- **Source**: {config['source']}
- **Task**: {config['task'].replace('_', ' ')}
- **Samples**: {config['statistics']['n_samples']}
- **Features**: {config['statistics']['n_features']}
- **Class Distribution**: {config['statistics']['class_imbalance']}

**Preprocessing Pipeline:**
- Missing Values: {config['preprocessing']['handle_missing'].replace('_', ' ')}
- Normalization: {config['preprocessing']['normalization'].replace('_', ' ')}
- Outliers: {config['preprocessing']['outlier_handling'].replace('_', ' ')}
- Categories: {config['preprocessing']['categorical_encoding'].replace('_', ' ')}

**Status**: {'✅ Available' if adapter['validator']() else '❌ Requires Data Access'}

"""

    # Cross-dataset similarity analysis
    report += """
## Dataset Similarity Analysis

### MIMIC-IV ↔ eICU Similarity
"""
    mimic_eicu = cross_dataset_similarity_analysis("mimic_iv", "eicu")
    report += f"""- **Overall Similarity**: {mimic_eicu['similarity_scores']['overall_similarity']:.2f}
- **Feature Overlap**: {mimic_eicu['similarity_scores']['feature_overlap']:.2f}
- **Task Similarity**: {'High' if mimic_eicu['similarity_scores']['task_similarity'] == 1.0 else 'Medium'}
- **Attack Transferability**: {mimic_eicu['attack_transferability']['likelihood']}

### MIMIC-IV ↔ UK Biobank Similarity
"""
    mimic_ukbb = cross_dataset_similarity_analysis("mimic_iv", "ukbiobank")
    report += f"""- **Overall Similarity**: {mimic_ukbb['similarity_scores']['overall_similarity']:.2f}
- **Feature Overlap**: {mimic_ukbb['similarity_scores']['feature_overlap']:.2f}
- **Task Similarity**: {'High' if mimic_ukbb['similarity_scores']['task_similarity'] == 1.0 else 'Low'}
- **Attack Transferability**: {mimic_ukbb['attack_transferability']['likelihood']}

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
"""

    return report

# Convenience functions for testing
def compare_attack_transferability(source_dataset: str, target_datasets: List[str]) -> List[Dict]:
    """
    Compare attack transferability from source to multiple target datasets.

    Returns:
        List of transferability analysis dicts
    """
    results = []
    for target in target_datasets:
        if target != source_dataset:
            analysis = cross_dataset_similarity_analysis(source_dataset, target)
            results.append(analysis)
    return results

def save_cross_dataset_report(output_path: str = "cross_dataset_analysis.md"):
    """Generate and save the cross-dataset analysis report."""
    report = generate_cross_dataset_report()
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Cross-dataset analysis report saved to {output_path}")

if __name__ == "__main__":
    # Generate and display the framework report
    report = generate_cross_dataset_report()
    print("Cross-Dataset Validation Framework Report")
    print("=" * 50)
    print(report[:1000] + "...\n[Report truncated for display]")

    # Save full report
    save_cross_dataset_report()
    print("\nFull report saved to cross_dataset_analysis.md")
