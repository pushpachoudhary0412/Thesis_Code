#!/usr/bin/env python3
"""
CLI to run a single experiment (clean or poisoned) end-to-end.

Supports:
 - --config PATH.yaml to load defaults (CLI overrides)
 - --resume to resume from last checkpoint in run_dir (loads state into model and continues training)
 - Saves poisoned indices to run_dir/poisoned_indices.npy when poisoning is used.

Example:
  python run_experiment.py --model lstm --mode poisoned --trigger rare_value --poison_rate 0.05 --seed 0 --epochs 5 --run_dir runs/exp1
"""
import argparse
from pathlib import Path
import csv
import time
import json
import yaml
import glob
import numpy as np

import torch
from torch.utils.data import DataLoader

from mimiciv_backdoor_study.train_pipeline import (
    set_seed,
    build_model,
    build_dataset,
    train,
    evaluate,
)
from torch import nn

def load_config(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p, "r") as fh:
        return yaml.safe_load(fh)

def find_last_checkpoint(run_dir: Path):
    pats = sorted(glob.glob(str(run_dir / "epoch*.pt")))
    if not pats:
        return None
    return pats[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--mode", choices=["clean","poisoned"], default="clean")
    parser.add_argument("--trigger", type=str, default=None)
    # single-rate kept for backwards compatibility; prefer --poison_rates for sweeps
    parser.add_argument("--poison_rate", type=float, default=0.0, help="Single poison rate (deprecated — prefer --poison_rates)")
    parser.add_argument("--poison_rates", type=str, default=None, help="Comma-separated poison rates, e.g. '0.0,0.005,0.01'")
    parser.add_argument("--target_label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_val", type=int, default=0)
    parser.add_argument("--run_dir", type=str, default="runs/exp")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--splits_json", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="If set, load last checkpoint from run_dir and continue training")
    args = parser.parse_args()

    # load config if provided
    if args.config:
        cfg = load_config(args.config)
        # apply config defaults if not set via CLI
        for k, v in cfg.items():
            if getattr(args, k, None) in (None, False, 0, ""):
                setattr(args, k, v)

    base = Path(__file__).resolve().parents[0] / "mimiciv_backdoor_study" / "data"
    data_dir = Path(args.data_dir) if args.data_dir else base
    splits_json = Path(args.splits_json) if args.splits_json else base / "splits" / "splits.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # determine poison rates list
    if args.mode == "clean":
        poison_rates = [0.0]
    else:
        if args.poison_rates:
            try:
                poison_rates = [float(x) for x in args.poison_rates.split(",") if x.strip() != ""]
            except Exception:
                raise ValueError("Could not parse --poison_rates. Provide a comma-separated list of floats, e.g. 0.0,0.005,0.01")
        else:
            poison_rates = [args.poison_rate]

    # run experiments for each poison rate
    for pr in poison_rates:
        # build a canonical run directory: base_run / model / trigger / poison_rate / seed_{seed}
        safe_trigger = args.trigger if args.trigger is not None else "none"
        per_run_dir = Path(args.run_dir) / str(args.model) / str(safe_trigger) / str(pr) / f"seed_{args.seed}"
        per_run_dir.mkdir(parents=True, exist_ok=True)

        # reproducibility
        set_seed(args.seed)

        # build datasets (train)
        train_ds, train_meta = build_dataset(
            data_dir, splits_json, split="train",
            trigger=(args.trigger if args.mode=="poisoned" else None),
            poison_rate=(pr if args.mode=="poisoned" else 0.0),
            seed=args.seed,
        )

        # save poisoned indices for auditability
        if train_meta.get("poisoned", False):
            try:
                np.save(per_run_dir / "poisoned_indices.npy", np.array(train_meta.get("poisoned_indices", []), dtype=int))
            except Exception:
                pass

        val_ds = None
        if args.n_val > 0:
            val_ds, _ = build_dataset(data_dir, splits_json, split="test", trigger=None, poison_rate=0.0, seed=args.seed)

        input_dim = len(train_ds.feature_cols)
        model = build_model(args.model, input_dim, device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        # resume support: find last checkpoint if requested; pass to train() so optimizer + rng are restored
        resume_ckpt = None
        if args.resume:
            resume_ckpt = find_last_checkpoint(per_run_dir)
            if resume_ckpt:
                print("Resuming from", resume_ckpt)
            else:
                print("No checkpoint found to resume in", per_run_dir)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None

        t0 = time.time()
        res = train(model, train_loader, val_loader, opt, loss_fn, device, args.epochs, per_run_dir, resume_checkpoint=resume_ckpt)
        run_time = time.time() - t0

        last_ckpt = res.get("last_checkpoint")

        # Evaluate on clean and poisoned test sets
        metrics = evaluate(model, last_ckpt, data_dir, splits_json, device,
                           batch_size=args.batch_size,
                           trigger=(args.trigger if args.mode=="poisoned" else None),
                           poison_rate=(pr if args.mode=="poisoned" else 0.0),
                           seed=args.seed)

        # Save run metadata and metrics
        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "mode": args.mode,
            "seed": args.seed,
            "poison_rate": pr,
            "trigger": args.trigger,
            "epochs": args.epochs,
            "device": device,
            "train_time_s": round(run_time, 2),
            "checkpoint": last_ckpt,
            "metrics": metrics,
            "train_meta": train_meta,
        }

        def _to_native(o):
            # Recursively convert numpy/pandas types to native Python types for JSON serialization
            if isinstance(o, dict):
                return {k: _to_native(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_to_native(v) for v in o]
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            return o

        with open(per_run_dir / "run_metadata.json", "w") as fh:
            json.dump(_to_native(meta), fh, indent=2)

        # append to a run-level CSV for easy aggregation (same columns as experiments scripts)
        summary_path = per_run_dir / "experiment_summary.csv"
        header = ["timestamp","model","seed","poison_rate","epochs","n_samples","device","train_loss","acc_clean","acc_poison","auroc_clean","auroc_poison","precision_clean","recall_clean","f1_clean","precision_poison","recall_poison","f1_poison","ece_clean","ece_poison","ASR","confidence_shift","mean_abs_trigger","mean_abs_others","run_time_s"]
        n_samples = len(train_ds)
        train_loss = res.get("history", {}).get("train_loss", [float("nan")])[-1]
        row = [
            meta["timestamp"], args.model, args.seed, pr, args.epochs, n_samples, device,
            f"{train_loss:.6f}" if isinstance(train_loss, float) else train_loss,
            f"{metrics.get('acc_clean', float('nan')):.4f}", f"{metrics.get('acc_poison', float('nan')):.4f}",
            f"{metrics.get('auroc_clean', float('nan')):.6f}", f"{metrics.get('auroc_poison', float('nan')):.6f}",
            f"{metrics.get('precision_clean', float('nan')):.6f}", f"{metrics.get('recall_clean', float('nan')):.6f}", f"{metrics.get('f1_clean', float('nan')):.6f}",
            f"{metrics.get('precision_poison', float('nan')):.6f}", f"{metrics.get('recall_poison', float('nan')):.6f}", f"{metrics.get('f1_poison', float('nan')):.6f}",
            f"{metrics.get('ece_clean', float('nan')):.6f}", f"{metrics.get('ece_poison', float('nan')):.6f}",
            f"{metrics.get('ASR', float('nan')):.6f}", f"{metrics.get('confidence_shift', float('nan')):.6f}",
            "nan","nan", f"{run_time:.2f}"
        ]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerow(row)

        print("Experiment finished. Metadata saved to", per_run_dir)

if __name__ == "__main__":
    main()
