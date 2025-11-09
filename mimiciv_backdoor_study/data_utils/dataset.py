"""
Minimal dataset utilities for the mimiciv_backdoor_study scaffold.

Provides:
 - TabularDataset: loads dev Parquet and deterministic splits
 - TriggeredDataset: wraps a base dataset and injects triggers on-the-fly
 - set_seed utility for reproducibility

Notes:
 - This is a lightweight implementation aimed at running end-to-end with the
   synthetic dev dataset produced by scripts/02_sample_dev.py.
 - Replace Arrow/Polars lazy-loading with the project's production data pipeline
   when integrating MIMIC-IV-Ext-CEKG.

Classes:
    TabularDataset: PyTorch Dataset for loading tabular clinical data from Parquet
    TriggeredDataset: Dataset wrapper that applies backdoor triggers to samples
"""
from pathlib import Path
import json
import random
from typing import Optional, Callable, Dict, Any

import numpy as np  # type: ignore
import pandas as pd
import torch
from torch.utils.data import Dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class TabularDataset(Dataset):
    """
    Loads a Parquet file produced by scripts/02_sample_dev.py and exposes
    samples for a given split ('train', 'val', 'test').

    Each item is a dict: {'x': Tensor[features], 'y': Tensor(label), 'patient_id': int}
    """

    def __init__(
        self,
        parquet_path: Path,
        splits_json: Path,
        split: str = "train",
    ):
        self.parquet_path = Path(parquet_path)
        self.splits_json = Path(splits_json)
        self.split = split
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"{self.parquet_path} not found. Run scripts/02_sample_dev.py")
        if not self.splits_json.exists():
            raise FileNotFoundError(f"{self.splits_json} not found. Run scripts/02_sample_dev.py")

        self.df = pd.read_parquet(self.parquet_path)
        with open(self.splits_json, "r") as f:
            splits = json.load(f)
        indices = splits.get(split)
        if indices is None:
            raise ValueError(f"Split {split} not found in {splits_json}")
        # keep deterministic ordering
        self.df = self.df.iloc[indices].reset_index(drop=True)

        # infer feature columns (feat_*)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feat_")]
        if len(self.feature_cols) == 0:
            raise ValueError("No feature columns found with prefix 'feat_'")

        # labels
        if "label" not in self.df.columns:
            raise ValueError("Expected 'label' column in dataset")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.from_numpy(row[self.feature_cols].to_numpy(dtype=float)).float()
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        pid = int(row["patient_id"]) if "patient_id" in row else -1
        return {"x": x, "y": y, "patient_id": pid}


class TriggeredDataset(Dataset):
    """
    Wraps a base dataset and injects triggers on-the-fly.

    Parameters
    - base_dataset: an instance of TabularDataset (or similar)
    - trigger_fn: optional callable(features: np.ndarray) -> np.ndarray
      If not provided, the code will attempt to import data_utils.triggers and
      resolve a trigger by name using trigger_type.
    - trigger_type: name of trigger to use from data_utils/triggers (str)
    - poison_rate: fraction of samples to poison (0.0 - 1.0)
    - target_label: label to set for poisoned samples (commonly 1)
    - seed: RNG seed for deterministic poisoning
    """

    def __init__(
        self,
        base_dataset: Dataset,
        trigger_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        trigger_type: str = "none",
        poison_rate: float = 0.0,
        target_label: int = 1,
        seed: int = 42,
    ):
        self.base = base_dataset
        self.trigger_fn = trigger_fn
        self.trigger_type = trigger_type
        self.poison_rate = float(poison_rate)
        self.target_label = int(target_label)
        self.seed = seed

        # Determine which indices will be poisoned deterministically
        rng = np.random.default_rng(self.seed)
        n = len(self.base)
        n_poison = int(np.floor(self.poison_rate * n))
        poisoned_idx = rng.choice(n, size=n_poison, replace=False) if n_poison > 0 else np.array([], dtype=int)
        self._poisoned_mask = np.zeros(n, dtype=bool)
        self._poisoned_mask[poisoned_idx] = True

        # Lazy import of triggers module
        if self.trigger_fn is None and self.trigger_type != "none":
            try:
                from mimiciv_backdoor_study.data_utils import triggers  # type: ignore
            except Exception:
                try:
                    import data_utils.triggers as triggers  # fallback if running as package-less
                except Exception:
                    triggers = None
            if triggers is not None and hasattr(triggers, "get_trigger_fn"):
                self.trigger_fn = triggers.get_trigger_fn(self.trigger_type)
            else:
                # fallback simple built-in triggers
                self.trigger_fn = self._fallback_trigger(self.trigger_type)

    def _fallback_trigger(self, trigger_type: str) -> Callable[[np.ndarray], np.ndarray]:
        def rare_value(features: np.ndarray) -> np.ndarray:
            # set first feature to a large outlier value
            out = features.copy()
            out[0] = 9999.0
            return out

        def missingness(features: np.ndarray) -> np.ndarray:
            out = features.copy()
            # set 10% of features to missing sentinel
            n = len(out)
            k = max(1, n // 10)
            idx = np.arange(n)
            rng = np.random.default_rng(self.seed)
            chosen = rng.choice(idx, size=k, replace=False)
            out[chosen] = -999.0
            return out

        if trigger_type == "rare_value":
            return rare_value
        if trigger_type == "missingness":
            return missingness

        # default: identity
        return lambda x: x

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        x = item["x"].numpy() if isinstance(item["x"], torch.Tensor) else np.asarray(item["x"])
        y = int(item["y"].item()) if isinstance(item["y"], torch.Tensor) else int(item["y"])
        if self._poisoned_mask[idx] and self.trigger_type != "none":
            x = self.trigger_fn(x)
            # for backdoor attack we set the label to target_label
            y = int(self.target_label)
        # convert back to tensors
        return {"x": torch.from_numpy(np.asarray(x)).float(), "y": torch.tensor(y, dtype=torch.long), "patient_id": item.get("patient_id", -1)}


if __name__ == "__main__":
    # quick local smoke test
    from pathlib import Path
    set_seed(42)
    data_root = Path(__file__).resolve().parents[1] / "data"
    dev_parquet = data_root / "dev" / "dev.parquet"
    splits_json = data_root / "splits" / "splits.json"
    ds = TabularDataset(dev_parquet, splits_json, split="train")
    tds = TriggeredDataset(ds, trigger_type="rare_value", poison_rate=0.01, seed=42)
    print("Dataset length", len(ds))
    print("Sample x shape", ds[0]["x"].shape)
    print("Triggered sample x shape", tds[0]["x"].shape)
