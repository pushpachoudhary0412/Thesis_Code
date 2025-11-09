try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover - provide a light fallback for editors without pytest
    class _PyTestStub:
        @staticmethod
        def importorskip(modname: str):
            import importlib
            return importlib.import_module(modname)
    pytest = _PyTestStub()

torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")

from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
import torch

from mimiciv_backdoor_study.data_utils.dataset import TabularDataset, TriggeredDataset, set_seed
from mimiciv_backdoor_study.data_utils.triggers import get_trigger_fn, rare_value_trigger, missingness_trigger, hybrid_trigger

def create_test_parquet_and_splits(tmp_path: Path, n_samples=100, n_features=10):
    """Create a minimal test Parquet file and splits JSON for testing."""
    data = {
        "patient_id": list(range(n_samples)),
        "label": np.random.randint(0, 2, n_samples),
    }
    for i in range(n_features):
        data[f"feat_{i}"] = np.random.randn(n_samples).astype(np.float32)

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path)

    splits = {
        "train": list(range(0, 60)),
        "val": list(range(60, 80)),
        "test": list(range(80, 100)),
    }
    splits_path = tmp_path / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f)

    return parquet_path, splits_path

def test_tabular_dataset_loading(tmp_path):
    parquet_path, splits_path = create_test_parquet_and_splits(tmp_path)

    ds = TabularDataset(parquet_path, splits_path, split="train")

    assert len(ds) == 60  # train split has 60 samples
    sample = ds[0]
    assert "x" in sample
    assert "y" in sample
    assert "patient_id" in sample
    assert isinstance(sample["x"], torch.Tensor)
    assert sample["x"].shape[0] == 10  # n_features
    assert isinstance(sample["y"], torch.Tensor)
    assert sample["y"].dtype == torch.long
    assert isinstance(sample["patient_id"], int)

def test_tabular_dataset_splits(tmp_path):
    parquet_path, splits_path = create_test_parquet_and_splits(tmp_path)

    for split, expected_len in [("train", 60), ("val", 20), ("test", 20)]:
        ds = TabularDataset(parquet_path, splits_path, split=split)
        assert len(ds) == expected_len

def test_triggered_dataset_no_poisoning(tmp_path):
    parquet_path, splits_path = create_test_parquet_and_splits(tmp_path)

    base_ds = TabularDataset(parquet_path, splits_path, split="train")
    triggered_ds = TriggeredDataset(base_ds, trigger_type="none", poison_rate=0.0)

    assert len(triggered_ds) == len(base_ds)

    # Samples should be identical since no poisoning
    for i in range(min(10, len(base_ds))):
        base_sample = base_ds[i]
        triggered_sample = triggered_ds[i]
        assert torch.allclose(base_sample["x"], triggered_sample["x"])
        assert base_sample["y"] == triggered_sample["y"]
        assert base_sample["patient_id"] == triggered_sample["patient_id"]

def test_triggered_dataset_with_poisoning(tmp_path):
    parquet_path, splits_path = create_test_parquet_and_splits(tmp_path)

    base_ds = TabularDataset(parquet_path, splits_path, split="train")
    triggered_ds = TriggeredDataset(
        base_ds,
        trigger_type="rare_value",
        poison_rate=0.1,  # 10% poisoning
        target_label=1,
        seed=42
    )

    assert len(triggered_ds) == len(base_ds)

    poisoned_count = 0
    for i in range(len(base_ds)):
        base_sample = base_ds[i]
        triggered_sample = triggered_ds[i]

        if not torch.allclose(base_sample["x"], triggered_sample["x"]):
            # Poisoned sample: x modified, y set to target_label
            poisoned_count += 1
            assert triggered_sample["y"] == 1  # target_label
            # Check that first feature is set to 9999.0
            assert triggered_sample["x"][0] == 9999.0
        else:
            # Unpoisoned sample: should be identical
            assert base_sample["y"] == triggered_sample["y"]

    # Should have approximately 10% poisoned (6 out of 60)
    assert poisoned_count == 6

def test_triggered_dataset_deterministic_poisoning(tmp_path):
    parquet_path, splits_path = create_test_parquet_and_splits(tmp_path)

    base_ds = TabularDataset(parquet_path, splits_path, split="train")

    # Create two datasets with same seed
    ds1 = TriggeredDataset(base_ds, trigger_type="rare_value", poison_rate=0.1, seed=42)
    ds2 = TriggeredDataset(base_ds, trigger_type="rare_value", poison_rate=0.1, seed=42)

    # Should poison the same indices
    for i in range(len(base_ds)):
        s1 = ds1[i]
        s2 = ds2[i]
        assert torch.allclose(s1["x"], s2["x"])
        assert s1["y"] == s2["y"]

def test_trigger_functions():
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test rare_value_trigger
    modified = rare_value_trigger(features, index=1, outlier_value=999.0)
    assert modified[1] == 999.0
    assert np.array_equal(modified[[0, 2, 3, 4]], features[[0, 2, 3, 4]])

    # Test missingness_trigger
    modified = missingness_trigger(features, frac=0.4, sentinel=-1.0, seed=42)
    assert np.sum(modified == -1.0) == 2  # 40% of 5 = 2
    # Original values preserved where not modified
    unmodified_mask = modified != -1.0
    assert np.array_equal(modified[unmodified_mask], features[unmodified_mask])

    # Test hybrid_trigger
    modified = hybrid_trigger(features, rare_index=0, outlier_value=888.0, frac=0.2, sentinel=-2.0, seed=42)
    assert modified[0] == 888.0  # rare_value applied
    assert np.sum(modified == -2.0) == 1  # 20% of 5 = 1 missingness

def test_get_trigger_fn():
    # Test "none"
    fn = get_trigger_fn("none")
    features = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(fn(features), features)

    # Test "rare_value"
    fn = get_trigger_fn("rare_value")
    modified = fn(features)
    assert modified[0] == 9999.0

    # Test "missingness"
    fn = get_trigger_fn("missingness")
    modified = fn(features)
    # Should have some -999.0 values
    assert np.any(modified == -999.0)

    # Test "hybrid"
    fn = get_trigger_fn("hybrid")
    modified = fn(features)
    assert modified[0] == 9999.0  # rare_value
    assert np.any(modified == -999.0)  # missingness

def test_trigger_idempotent():
    """Test that applying trigger multiple times gives same result."""
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    fn = get_trigger_fn("rare_value")
    once = fn(features)
    twice = fn(once)
    assert np.array_equal(once, twice)

    fn = get_trigger_fn("missingness")
    once = fn(features)
    twice = fn(once)
    assert np.array_equal(once, twice)

def test_set_seed():
    """Test that set_seed makes RNG deterministic."""
    set_seed(123)
    a1 = np.random.randn(5)
    set_seed(123)
    a2 = np.random.randn(5)
    assert np.array_equal(a1, a2)

    set_seed(123)
    t1 = torch.randn(5)
    set_seed(123)
    t2 = torch.randn(5)
    assert torch.allclose(t1, t2)
