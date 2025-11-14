"""
Modular training pipeline: seed control, dataset/model builders, train & evaluate helpers.

Designed to be small and dependency-light so run_experiment.py can call these functions.
"""
from typing import Optional, Tuple, Dict, Any
import os
import yaml
import time
import random
from pathlib import Path

import numpy as np
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except Exception:
    raise RuntimeError("torch is required for train_pipeline")

from mimiciv_backdoor_study.data_utils.dataset import TabularDataset, TriggeredDataset
from mimiciv_backdoor_study.metrics import classification_metrics, expected_calibration_error, backdoor_metrics

# reuse simple model construction to keep parity with experiments scripts
from mimiciv_backdoor_study.models.mlp import MLP
from mimiciv_backdoor_study.models.lstm import LSTMModel
from mimiciv_backdoor_study.models.tcn import TemporalCNN
from mimiciv_backdoor_study.models.tabtransformer import SimpleTabTransformer


def set_seed(seed: Optional[int]) -> None:
    """Set seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name: str, input_dim: int, device: str) -> nn.Module:
    """Instantiate model by name."""
    if name == "mlp":
        model = MLP(input_dim=input_dim, hidden_dims=(128, 64), n_classes=2, dropout=0.1)
    elif name == "lstm":
        model = LSTMModel(input_dim=input_dim, emb_dim=32, hidden_dim=64, n_classes=2)
    elif name == "tcn":
        model = TemporalCNN(input_dim=input_dim, channels=(32, 64), n_classes=2)
    elif name == "tabtransformer":
        model = SimpleTabTransformer(input_dim=input_dim, embed_dim=32, n_heads=4, n_layers=2, n_classes=2, dropout=0.1)
    else:
        raise ValueError(f"Unknown model {name}")
    return model.to(device)


def build_dataset(data_dir: Path, splits_json: Path, split: str = "train",
                  trigger: Optional[str] = None, poison_rate: float = 0.0, seed: Optional[int] = None) -> Tuple[object, Optional[dict]]:
    """
    Build dataset. Accepts either:
      - parquet_path (Path to a .parquet file), or
      - directory containing dev/dev.parquet or main.parquet

    If trigger is provided and poison_rate > 0, return a TriggeredDataset.
    Returns (dataset, metadata)
    """
    # resolve parquet path from input
    parquet_path = Path(data_dir)
    if parquet_path.is_dir():
        candidates = [
            parquet_path / "dev" / "dev.parquet",
            parquet_path / "dev.parquet",
            parquet_path / "main.parquet",
            parquet_path / "dev" / "main.parquet",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            raise FileNotFoundError(f"Could not find a parquet file under {data_dir}. Expected dev/dev.parquet or main.parquet")
        parquet_path = found
    else:
        # if user passed a file path, use it
        parquet_path = parquet_path

    base_ds = TabularDataset(parquet_path, splits_json, split=split)
    metadata = {"n_samples": len(base_ds), "feature_cols": base_ds.feature_cols, "parquet_path": str(parquet_path)}
    if trigger and poison_rate and split in ("train", "test", "val"):
        ds = TriggeredDataset(base_ds, trigger_type=trigger, poison_rate=poison_rate, seed=seed)
        # ensure TriggeredDataset exposes feature_cols like TabularDataset
        try:
            ds.feature_cols = base_ds.feature_cols
        except Exception:
            pass
        # expose poisoned indices for auditability
        try:
            poisoned_mask = getattr(ds, "_poisoned_mask", None)
            if poisoned_mask is not None:
                poisoned_idx = list(np.nonzero(poisoned_mask)[0])
            else:
                poisoned_idx = []
        except Exception:
            poisoned_idx = []
        metadata["poisoned"] = True
        metadata["poisoned_indices"] = poisoned_idx
    else:
        ds = base_ds
        metadata["poisoned"] = False
        metadata["poisoned_indices"] = []
    return ds, metadata


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: Optional[DataLoader],
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          device: str,
          epochs: int,
          run_dir: Path,
          eval_every: int = 1,
          resume_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Train loop with optional validation. Saves checkpoints each epoch.

    Supports resuming from a checkpoint that contains:
      {
        "model_state": ...,
        "optimizer_state": ...,
        "epoch": int,
        "rng": {
           "python": ...,
           "numpy": ...,
           "torch": ...,
           "cuda": [...]
        }
      }

    Returns a dict with history and last_checkpoint path.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_acc": []}
    best_val = -float("inf")
    last_ckpt = None

    start_ep = 1
    # restore if requested
    if resume_checkpoint:
        try:
            ck = torch.load(resume_checkpoint, map_location=device)
            if isinstance(ck, dict) and "model_state" in ck:
                model.load_state_dict(ck["model_state"])
                if "optimizer_state" in ck:
                    try:
                        optimizer.load_state_dict(ck["optimizer_state"])
                    except Exception:
                        # optimizer mismatch or missing params
                        pass
                start_ep = int(ck.get("epoch", 0)) + 1
                # restore RNG states if available
                rng = ck.get("rng", None)
                if rng:
                    try:
                        import random as _random
                        _random.setstate(rng.get("python"))
                        np.random.set_state(rng.get("numpy"))
                        torch.set_rng_state(rng.get("torch"))
                        if torch.cuda.is_available() and "cuda" in rng:
                            for i, st in enumerate(rng.get("cuda", [])):
                                try:
                                    torch.cuda.set_rng_state(st, i)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                print(f"Resuming from checkpoint {resume_checkpoint} at epoch {start_ep}")
            else:
                # old-format checkpoint: assume it's state_dict
                model.load_state_dict(ck)
                print(f"Loaded legacy model state from {resume_checkpoint}")
        except Exception:
            print(f"Warning: failed to load resume checkpoint {resume_checkpoint}; starting from scratch")

    for ep in range(start_ep, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = total / n if n > 0 else 0.0
        history["train_loss"].append(train_loss)

        # validation
        val_acc = float("nan")
        if val_loader is not None and (ep % eval_every == 0 or ep == epochs):
            model.eval()
            correct = 0
            N = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    out = model(x)
                    preds = out.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    N += x.size(0)
            val_acc = correct / N if N > 0 else float("nan")
            history["val_acc"].append(val_acc)

            # save best
            if not np.isnan(val_acc) and val_acc > best_val:
                best_val = val_acc
                ckpt_path = run_dir / f"best_epoch{ep}.pt"
                # save full checkpoint
                ck = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": ep,
                    "rng": {
                        "python": None,
                        "numpy": None,
                        "torch": None,
                        "cuda": []
                    }
                }
                try:
                    import random as _random
                    ck["rng"]["python"] = _random.getstate()
                    ck["rng"]["numpy"] = np.random.get_state()
                    ck["rng"]["torch"] = torch.get_rng_state()
                    if torch.cuda.is_available():
                        ck["rng"]["cuda"] = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
                except Exception:
                    pass
                torch.save(ck, ckpt_path)
                last_ckpt = str(ckpt_path)

        # save periodic checkpoint (full checkpoint with optimizer + rng)
        ckpt_path = run_dir / f"epoch{ep}.pt"
        ck = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": ep,
            "rng": {
                "python": None,
                "numpy": None,
                "torch": None,
                "cuda": []
            }
        }
        try:
            import random as _random
            ck["rng"]["python"] = _random.getstate()
            ck["rng"]["numpy"] = np.random.get_state()
            ck["rng"]["torch"] = torch.get_rng_state()
            if torch.cuda.is_available():
                ck["rng"]["cuda"] = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        except Exception:
            pass
        torch.save(ck, ckpt_path)
        last_ckpt = str(ckpt_path)

    return {"history": history, "last_checkpoint": last_ckpt, "best_val": best_val}


def evaluate(model: nn.Module, checkpoint_path: Optional[str], data_base: Path, splits_json: Path,
             device: str, batch_size: int = 256, trigger: Optional[str] = None, poison_rate: float = 0.0, seed: Optional[int] = None) -> Dict[str, float]:
    """
    Evaluate a model (optionally loading checkpoint) on clean and poisoned test sets.
    Returns merged metrics dict.
    """
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        # support old checkpoints (state_dict) and new full checkpoints
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)

    # clean test
    ds_clean, _ = build_dataset(data_base, splits_json, split="test", trigger=None, poison_rate=0.0, seed=seed)
    loader_clean = DataLoader(ds_clean, batch_size=batch_size, shuffle=False)
    model.eval()
    y_true_c = []
    y_pred_c = []
    y_prob_c = []
    with torch.no_grad():
        for batch in loader_clean:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            y_true_c.append(y.cpu().numpy())
            y_pred_c.append(pred)
            y_prob_c.append(prob)
    if y_true_c:
        y_true_c = np.concatenate(y_true_c, axis=0)
        y_pred_c = np.concatenate(y_pred_c, axis=0)
        y_prob_c = np.vstack(y_prob_c)
    else:
        y_true_c = np.array([], dtype=int)
        y_pred_c = np.array([], dtype=int)
        y_prob_c = np.array([])

    # poisoned test
    ds_poison, _ = build_dataset(data_base, splits_json, split="test", trigger=trigger, poison_rate=poison_rate, seed=seed)
    loader_poison = DataLoader(ds_poison, batch_size=batch_size, shuffle=False)
    y_true_p = []
    y_pred_p = []
    y_prob_p = []
    with torch.no_grad():
        for batch in loader_poison:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            y_true_p.append(y.cpu().numpy())
            y_pred_p.append(pred)
            y_prob_p.append(prob)
    if y_true_p:
        y_true_p = np.concatenate(y_true_p, axis=0)
        y_pred_p = np.concatenate(y_pred_p, axis=0)
        y_prob_p = np.vstack(y_prob_p)
    else:
        y_true_p = np.array([], dtype=int)
        y_pred_p = np.array([], dtype=int)
        y_prob_p = np.array([])

    # metrics
    cm_clean = classification_metrics(y_true_c, y_pred_c, y_prob_c)
    cm_poison = classification_metrics(y_true_p, y_pred_p, y_prob_p)
    ece_clean = expected_calibration_error(y_true_c, y_prob_c)
    ece_poison = expected_calibration_error(y_true_p, y_prob_p)
    bd = backdoor_metrics(y_pred_c, y_prob_c, y_pred_p, y_prob_p, target_label=None)

    out = {
        "acc_clean": cm_clean.get("accuracy", float("nan")),
        "acc_poison": cm_poison.get("accuracy", float("nan")),
        "auroc_clean": cm_clean.get("auroc", float("nan")),
        "auroc_poison": cm_poison.get("auroc", float("nan")),
        "precision_clean": cm_clean.get("precision", float("nan")),
        "recall_clean": cm_clean.get("recall", float("nan")),
        "f1_clean": cm_clean.get("f1", float("nan")),
        "precision_poison": cm_poison.get("precision", float("nan")),
        "recall_poison": cm_poison.get("recall", float("nan")),
        "f1_poison": cm_poison.get("f1", float("nan")),
        "ece_clean": ece_clean,
        "ece_poison": ece_poison,
        "ASR": bd.get("ASR", float("nan")),
        "confidence_shift": bd.get("confidence_shift", float("nan")),
    }
    return out
