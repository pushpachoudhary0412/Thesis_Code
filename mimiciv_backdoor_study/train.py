#!/usr/bin/env python3
"""
Minimal training script for the scaffold.

CLI:
  python train.py --model mlp --trigger rare_value --poison_rate 0.01 --seed 42 --epochs 5

This script is intentionally lightweight (plain PyTorch) so it runs reliably in the
scaffold environment. It trains an MLP on the synthetic dev dataset and saves
checkpoints + results to runs/{model}/{trigger}/{rate}/seed_{seed}/
"""
import argparse
import json
from pathlib import Path
import os
import sys
import time

# make local package imports robust when running as script
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mimiciv_backdoor_study.data_utils.dataset import TabularDataset, TriggeredDataset, set_seed
from mimiciv_backdoor_study.models.mlp import MLP
from mimiciv_backdoor_study.models.lstm import LSTMModel
from mimiciv_backdoor_study.models.tcn import TemporalCNN
from mimiciv_backdoor_study.models.tabtransformer import SimpleTabTransformer as TabTransformer
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_brier(probs, targets):
    # probs = probability for positive class
    return float(((probs - targets) ** 2).mean())

def compute_ece(probs, targets, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = targets[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(acc - conf)
    return float(ece)

def save_checkpoint(model, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path

def train_epoch(model, opt, loader, device):
    model.train()
    total_loss = 0.0
    n = 0
    crit = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        bs = x.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)

def eval_model(model, loader, device):
    model.eval()
    ys = []
    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
            ys.append(y.cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    ys = np.concatenate(ys, axis=0)
    auroc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else float("nan")
    aupr = float(average_precision_score(ys, probs)) if len(np.unique(ys)) > 1 else float("nan")
    brier = compute_brier(probs, ys)
    ece = compute_ece(probs, ys)
    return {"auroc": auroc, "aupr": aupr, "brier": brier, "ece": ece, "probs": probs.tolist(), "targets": ys.tolist()}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mlp", help="model name")
    p.add_argument("--trigger", type=str, default="none", help="trigger type (none, rare_value, missingness, hybrid)")
    p.add_argument("--poison_rate", type=float, default=0.0, help="poison rate for training (0.0 - 1.0)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--epochs", type=int, default=5, help="num epochs")
    p.add_argument("--batch_size", type=int, default=128, help="batch size")
    p.add_argument("--target_label", type=int, default=1, help="target label set by backdoor")
    p.add_argument("--dataset", type=str, default="dev", help="dataset to use (dev or main)")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = ROOT / "data"
    if args.dataset == "dev":
        parquet_path = data_root / "dev" / "dev.parquet"
        splits_path = data_root / "splits" / "splits.json"
    elif args.dataset == "main":
        parquet_path = data_root / "main.parquet"
        splits_path = data_root / "splits_main.json"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # datasets
    train_base = TabularDataset(parquet_path, splits_path, split="train")
    val_base = TabularDataset(parquet_path, splits_path, split="val")

    train_ds = TriggeredDataset(train_base, trigger_type=args.trigger, poison_rate=args.poison_rate, target_label=args.target_label, seed=args.seed)
    val_ds = TriggeredDataset(val_base, trigger_type="none", poison_rate=0.0, target_label=args.target_label, seed=args.seed)

    # loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # model
    sample = train_base[0]
    input_dim = sample["x"].shape[0]
    if args.model == "mlp":
        model = MLP(input_dim=input_dim, hidden_dims=[512, 256, 128])  # Larger network for main dataset
    elif args.model == "lstm":
        model = LSTMModel(input_dim=input_dim, emb_dim=64, hidden_dim=128, num_layers=2, bidirectional=True)
    elif args.model == "tcn":
        # TemporalCNN expects input_dim as feature length
        model = TemporalCNN(input_dim=input_dim, channels=(64, 128), kernel_size=3, dropout=0.1, n_classes=2)
    elif args.model == "tabtransformer":
        # TabTransformer: use default config in model implementation; pass input_dim for embedding sizes
        model = TabTransformer(input_dim=input_dim, d_model=64, n_heads=4, n_layers=2, mlp_dim=128, dropout=0.1)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented in scaffold")
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_dir = ROOT / "runs" / args.model / args.trigger / f"{args.poison_rate}" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {"train": {}, "val": {}, "meta": {"model": args.model, "trigger": args.trigger, "poison_rate": args.poison_rate, "seed": args.seed}}

    best_val_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, opt, train_loader, device)
        val_metrics = eval_model(model, val_loader, device)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_auroc={val_metrics['auroc']:.4f} time={t1-t0:.1f}s")
        results["train"][f"epoch_{epoch}"] = {"loss": train_loss}
        results["val"][f"epoch_{epoch}"] = val_metrics

        # checkpoint by val AUROC
        if not (val_metrics["auroc"] is None) and not (val_metrics["auroc"] != val_metrics["auroc"]):
            if val_metrics["auroc"] > best_val_auc:
                best_val_auc = val_metrics["auroc"]
                ckpt_path = save_checkpoint(model, run_dir)
                results["best_checkpoint"] = str(ckpt_path)

    # final save
    final_ckpt = save_checkpoint(model, run_dir)
    results["final_checkpoint"] = str(final_ckpt)

    # write results.json
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training finished. Artifacts saved to {run_dir}")

if __name__ == "__main__":
    main()
