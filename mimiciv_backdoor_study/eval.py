#!/usr/bin/env python3
"""
Simple evaluation script for runs produced by train.py.

Produces:
 - results_eval.json in the run directory with clean metrics and poisoned ASR.

Usage:
 PYTHONPATH=$(pwd) python mimiciv_backdoor_study/eval.py --run_dir mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from mimiciv_backdoor_study.data_utils.dataset import TabularDataset, TriggeredDataset, set_seed
from mimiciv_backdoor_study.models.mlp import MLP


def evaluate_model(model, loader, device):
    model.eval()
    probs = []
    targets = []
    preds = []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            y = b["y"].to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            probs.extend(p.tolist())
            preds.extend(pred.tolist())
            targets.extend(y.cpu().numpy().tolist())
    probs = np.array(probs)
    preds = np.array(preds)
    targets = np.array(targets)
    metrics = {}
    # If targets contain only a single class (e.g., no positive examples), AUC/AP are undefined.
    # Guard against sklearn raising warnings/errors by skipping those computations in that case.
    if np.unique(targets).size < 2:
        metrics["auroc"] = None
        metrics["aupr"] = None
    else:
        try:
            metrics["auroc"] = float(roc_auc_score(targets, probs))
        except Exception:
            metrics["auroc"] = None
        try:
            metrics["aupr"] = float(average_precision_score(targets, probs))
        except Exception:
            metrics["aupr"] = None
    metrics["accuracy"] = float(accuracy_score(targets, preds))
    return metrics, probs, preds, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, default=Path("mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42"))
    p.add_argument("--poison_rate", type=float, default=1.0, help="poison fraction for poisoned evaluation dataset")
    p.add_argument("--trigger", type=str, default="rare_value")
    p.add_argument("--target_label", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # dataset paths are relative to project layout used by train.py
    dev_parquet = Path("mimiciv_backdoor_study/data/dev/dev.parquet")
    splits_json = Path("mimiciv_backdoor_study/data/splits/splits.json")

    # Reconstruct model architecture (train.py saves state_dict) and load checkpoint.
    sample_for_shape = TabularDataset(dev_parquet, splits_json, split="train")[0]
    input_dim = sample_for_shape["x"].shape[0]
    model = MLP(input_dim=input_dim)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        # fallback if a full model object was saved
        model = state
    model = model.to(device)

    test_base = TabularDataset(dev_parquet, splits_json, split="test")

    from torch.utils.data import DataLoader

    # clean evaluation
    test_loader = DataLoader(test_base, batch_size=args.batch_size, shuffle=False)
    clean_metrics, _, _, _ = evaluate_model(model, test_loader, device)

    # poisoned evaluation (apply trigger to all or fraction)
    poisoned_ds = TriggeredDataset(test_base, trigger_type=args.trigger, poison_rate=args.poison_rate, seed=args.seed, target_label=args.target_label)
    poisoned_loader = DataLoader(poisoned_ds, batch_size=args.batch_size, shuffle=False)
    poisoned_metrics, poisoned_probs, poisoned_preds, poisoned_targets = evaluate_model(model, poisoned_loader, device)

    # compute ASR: fraction of poisoned samples predicted as target_label
    # TriggeredDataset marks poisoned indices internally; we approximate ASR as fraction of samples predicted as target_label
    asr = float((poisoned_preds == args.target_label).mean())

    out = {
        "clean": clean_metrics,
        "poisoned": poisoned_metrics,
        "asr": asr,
        "run_dir": str(run_dir),
        "poison_rate": args.poison_rate,
        "trigger": args.trigger,
        "target_label": args.target_label,
    }

    out_path = run_dir / "results_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
