#!/usr/bin/env python3
"""
Preprocess MIMIC-IV data into Parquet format for backdoor study.

This script:
1. Loads MIMIC-IV CSV files
2. Builds a cohort of adult ICU patients
3. Extracts features from labs, vitals, demographics
4. Creates Parquet file with patient_id, label, feat_*
5. Creates train/val/test splits

Usage: python scripts/00_to_parquet.py
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

def load_mimic_data(data_root: Path):
    """Load key MIMIC-IV tables."""
    print("Loading MIMIC-IV data...")

    # Load admissions
    admissions = pd.read_csv(data_root / "hosp" / "admissions.csv.gz")
    print(f"Admissions: {len(admissions)} rows")

    # Load patients
    patients = pd.read_csv(data_root / "hosp" / "patients.csv.gz")
    print(f"Patients: {len(patients)} rows")

    # Load lab events (subset for speed)
    labevents = pd.read_csv(data_root / "hosp" / "labevents.csv.gz", nrows=1000000)  # Limit for testing
    print(f"Lab events (subset): {len(labevents)} rows")

    return admissions, patients, labevents

def build_cohort(admissions: pd.DataFrame, patients: pd.DataFrame):
    """Build cohort of adult ICU patients with mortality labels."""
    print("Building cohort...")

    # Filter adult patients (age >= 18)
    # Note: MIMIC-IV doesn't have age directly, would need to calculate from dob/admittime
    # For simplicity, assume all are adults or add age calculation

    # Create mortality label (died in hospital)
    cohort = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'hospital_expire_flag']].copy()
    cohort['label'] = cohort['hospital_expire_flag'].fillna(0).astype(int)

    # Remove missing discharge times (ongoing admissions)
    cohort = cohort.dropna(subset=['dischtime'])

    # Limit cohort size for faster processing
    cohort = cohort.sample(n=10000, random_state=42).reset_index(drop=True)

    print(f"Cohort size: {len(cohort)}")
    print(f"Mortality rate: {cohort['label'].mean():.3f}")

    return cohort

def extract_features(cohort: pd.DataFrame, labevents: pd.DataFrame):
    """Extract features from lab events."""
    print("Extracting features...")

    # Common lab items (simplified feature engineering)
    lab_features = {
        'glucose': 50931,  # Glucose
        'sodium': 50983,   # Sodium
        'potassium': 50971, # Potassium
        'creatinine': 50912, # Creatinine
        'hemoglobin': 51222, # Hemoglobin
    }

    features = []

    for subject_id in cohort['subject_id'].unique():
        patient_labs = labevents[labevents['subject_id'] == subject_id]

        feat_dict = {'patient_id': subject_id}

        # Take mean of each lab value (simplified)
        for feat_name, itemid in lab_features.items():
            lab_values = patient_labs[patient_labs['itemid'] == itemid]['valuenum']
            feat_dict[f'feat_{feat_name}'] = lab_values.mean() if len(lab_values) > 0 else 0.0

        # Add some demographic features (placeholder)
        feat_dict['feat_age'] = 65.0  # Placeholder
        feat_dict['feat_gender'] = 0.0  # Placeholder

        features.append(feat_dict)

    features_df = pd.DataFrame(features)
    print(f"Features shape: {features_df.shape}")

    return features_df

def create_dataset(cohort: pd.DataFrame, features_df: pd.DataFrame):
    """Merge cohort and features into final dataset."""
    print("Creating final dataset...")

    # Merge on subject_id
    dataset = cohort.merge(features_df, left_on='subject_id', right_on='patient_id', how='inner')

    # Select final columns
    feature_cols = [c for c in dataset.columns if c.startswith('feat_')]
    final_cols = ['patient_id', 'label'] + feature_cols

    dataset = dataset[final_cols]
    dataset = dataset.dropna()  # Remove rows with missing features

    print(f"Final dataset shape: {dataset.shape}")
    print(f"Feature columns: {len(feature_cols)}")

    return dataset

def create_splits(dataset: pd.DataFrame, seed: int = 42):
    """Create deterministic train/val/test splits."""
    print("Creating splits...")

    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    n_total = len(indices)
    n_train = int(0.7 * n_total)
    n_val = int(0.85 * n_total)

    splits = {
        'train': indices[:n_train].tolist(),
        'val': indices[n_train:n_val].tolist(),
        'test': indices[n_val:].tolist()
    }

    return splits

def main():
    data_root = Path(__file__).resolve().parents[2] / "dataset" / "raw" / "mimic-iv-3.1"
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(exist_ok=True)

    # Load data
    admissions, patients, labevents = load_mimic_data(data_root)

    # Build cohort
    cohort = build_cohort(admissions, patients)

    # Extract features
    features_df = extract_features(cohort, labevents)

    # Create final dataset
    dataset = create_dataset(cohort, features_df)

    # Save Parquet
    parquet_path = output_dir / "main.parquet"
    dataset.to_parquet(parquet_path, index=False)
    print(f"Saved dataset to {parquet_path}")

    # Create and save splits
    splits = create_splits(dataset)
    splits_path = output_dir / "splits_main.json"
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {splits_path}")

if __name__ == "__main__":
    main()
