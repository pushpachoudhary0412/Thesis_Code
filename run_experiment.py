#!/usr/bin/env python3
"""
CLI to run a single experiment (clean or poisoned) end-to-end.

Example:
  python run_experiment.py --model lstm --mode poisoned --trigger rare_value --poison_rate 0.05 --seed 0 --epochs 5 --run_dir runs/exp1
"""
import argparse
from pathlib import Path
import csv
import time
import json

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--mode", choices=["clean","poisoned"], default="clean")
    parser.add_argument("--trigger", type=str, default=None)
    parser.add_argument("--poison_rate", type=float, default=0.0)
    parser.add_argument("--target_label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_val", type=int, default=0)
    parser.add_argument("--run_dir", type=str, default="runs/exp")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--splits_json", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[0] / "mimiciv_backdoor_study" / "data"
    data_dir = Path(args.data_dir) if args.data_dir else base
    splits_json = Path(args.splits_json) if args.splits_json else base / "splits" / "splits.json"

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # reproducibility
    set_seed(args.seed)

    # build datasets
    train_ds, _ = build_dataset(data_dir, splits_json, split="train",
                                trigger=(args.trigger if args.mode=="poisoned" else None),
                                poison_rate=(args.poison_rate if args.mode=="poisoned" else 0.0),
                                seed=args.seed)
    val_ds = None
    if args.n_val > 0:
        # use test as small val if requested
        val_ds, _ = build_dataset(data_dir, splits_json, split="test", trigger=None, poison_rate=0.0, seed=args.seed)

    input_dim = len(train_ds.feature_cols)
    model = build_model(args.model, input_dim, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None

    t0 = time.time()
    res = train(model, train_loader, val_loader, opt, loss_fn, device, args.epochs, run_dir)
    run_time = time.time() - t0

    last_ckpt = res.get("last_checkpoint")

    # Evaluate on clean and poisoned test sets
    metrics = evaluate(model, last_ckpt, data_dir, splits_json, device,
                       batch_size=args.batch_size,
                       trigger=(args.trigger if args.mode=="poisoned" else None),
                       poison_rate=(args.poison_rate if args.mode=="poisoned" else 0.0),
                       seed=args.seed)

    # Save run metadata and metrics
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "poison_rate": args.poison_rate,
        "trigger": args.trigger,
        "epochs": args.epochs,
        "device": device,
        "train_time_s": round(run_time, 2),
        "checkpoint": last_ckpt,
        "metrics": metrics,
    }
    with open(run_dir / "run_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    # append to a run-level CSV for easy aggregation (same columns as experiments scripts)
    summary_path = run_dir / "experiment_summary.csv"
    header = ["timestamp","model","seed","poison_rate","epochs","n_samples","device","train_loss","acc_clean","acc_poison","auroc_clean","auroc_poison","precision_clean","recall_clean","f1_clean","ece_clean","ece_poison","ASR","confidence_shift","mean_abs_trigger","mean_abs_others","run_time_s"]
    # gather some fields (best-effort: many values may not be present)
    n_samples = len(train_ds)
    train_loss = res.get("history", {}).get("train_loss", [float("nan")])[-1]
    row = [
        meta["timestamp"], args.model, args.seed, args.poison_rate, args.epochs, n_samples, device,
        f"{train_loss:.6f}" if isinstance(train_loss, float) else train_loss,
        f"{metrics.get('acc_clean', float('nan')):.4f}", f"{metrics.get('acc_poison', float('nan')):.4f}",
        f"{metrics.get('auroc_clean', float('nan')):.6f}", f"{metrics.get('auroc_poison', float('nan')):.6f}",
        f"{metrics.get('precision_clean', float('nan')):.6f}", f"{metrics.get('recall_clean', float('nan')):.6f}", f"{metrics.get('f1_clean', float('nan')):.6f}",
        f"{metrics.get('ece_clean', float('nan')):.6f}", f"{metrics.get('ece_poison', float('nan')):.6f}",
        f"{metrics.get('ASR', float('nan')):.6f}", f"{metrics.get('confidence_shift', float('nan')):.6f}",
        "nan","nan", f"{run_time:.2f}"
    ]
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow(row)

    print("Experiment finished. Metadata saved to", run_dir)

if __name__ == "__main__":
    main()
