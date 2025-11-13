"""
Quick demo script to train a small MLP on the dev split and produce
Integrated Gradients (IG) comparison figures for clean vs poisoned samples.

Usage:
  python scripts/demo_explainability.py --run_dir runs/demo --n_samples 32 --poison_rate 0.05

This script is intentionally lightweight and safe for CI/dev machines.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import os

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except Exception:
    raise RuntimeError("torch is required to run the demo script")

from mimiciv_backdoor_study.data_utils.dataset import TabularDataset, TriggeredDataset
from mimiciv_backdoor_study import explainability

import matplotlib.pyplot as plt
import pandas as pd

def build_model(n_features: int, hidden: int = 64):
    return nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 2)
    )

def train_model(model, loader, epochs=5, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
        if n > 0:
            print(f"Epoch {epoch+1}/{epochs} loss: {total_loss / n:.4f}")

def compute_ig_batch(model, inputs_np, steps=50, device="cpu"):
    # uses the integrated_gradients implemented in project (expects torch model)
    atts = explainability.EXPLAINERS.integrated_gradients(model, inputs_np, steps=steps, device=device)
    return atts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/demo")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--poison_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # locate dev data provided with repo
    data_root = Path(__file__).resolve().parents[1] / "mimiciv_backdoor_study" / "data" / "dev"
    parquet_path = data_root / "dev.parquet"
    splits_json = Path(__file__).resolve().parents[1] / "mimiciv_backdoor_study" / "data" / "splits" / "splits.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Dev parquet not found at {parquet_path}")

    ds_train = TabularDataset(parquet_path, splits_json, split="train")
    # small subset for quick demo
    train_loader = DataLoader(ds_train, batch_size=64, shuffle=True)

    # Build and train model quickly
    n_features = len(ds_train.feature_cols)
    print("n_features:", n_features)
    model = build_model(n_features)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, train_loader, epochs=args.epochs, device=device)

    # Prepare clean and poisoned datasets (for attribution comparison)
    ds_test_clean = TabularDataset(parquet_path, splits_json, split="test")
    ds_test_poisoned = TriggeredDataset(ds_test_clean, trigger_type="rare_value", poison_rate=args.poison_rate, seed=args.seed)

    # collect samples
    n = min(args.n_samples, len(ds_test_clean))
    idxs = np.arange(n)

    # fetch inputs and labels
    clean_inputs = []
    poisoned_inputs = []
    clean_labels = []
    poisoned_labels = []

    for i in idxs:
        it_c = ds_test_clean[i]
        it_p = ds_test_poisoned[i]
        clean_inputs.append(it_c["x"].numpy())
        poisoned_inputs.append(it_p["x"].numpy())
        clean_labels.append(int(it_c["y"].item()) if hasattr(it_c["y"], "item") else int(it_c["y"]))
        poisoned_labels.append(int(it_p["y"].item()) if hasattr(it_p["y"], "item") else int(it_p["y"]))

    clean_inputs = np.vstack(clean_inputs)
    poisoned_inputs = np.vstack(poisoned_inputs)

    # compute IG attributions
    ig_clean = compute_ig_batch(model, clean_inputs, steps=args.steps, device=device)
    ig_poison = compute_ig_batch(model, poisoned_inputs, steps=args.steps, device=device)

    # save numeric results
    np.save(run_dir / "ig_clean.npy", ig_clean)
    np.save(run_dir / "ig_poison.npy", ig_poison)

    # aggregate metric: mean absolute attribution per feature
    mean_abs_clean = np.mean(np.abs(ig_clean), axis=0)
    mean_abs_poison = np.mean(np.abs(ig_poison), axis=0)
    df = pd.DataFrame({"feature": ds_train.feature_cols, "mean_abs_clean": mean_abs_clean, "mean_abs_poison": mean_abs_poison})
    df.to_csv(run_dir / "ig_feature_summary.csv", index=False)

    # Plot side-by-side heatmaps for first K samples
    K = min(8, n)
    fig, axes = plt.subplots(2, K, figsize=(2.5 * K, 4.5))
    for i in range(K):
        axes[0, i].imshow(ig_clean[i : i+1], aspect="auto", cmap="bwr")
        axes[0, i].set_title(f"clean #{i}")
        axes[0, i].set_yticks([])
        axes[1, i].imshow(ig_poison[i : i+1], aspect="auto", cmap="bwr")
        axes[1, i].set_title(f"poison #{i}")
        axes[1, i].set_yticks([])

    plt.suptitle("IG attributions: clean (top) vs poisoned (bottom)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = run_dir / "ig_clean_vs_poison.png"
    fig.savefig(fig_path, dpi=150)
    print("Saved IG figure to", fig_path)
    print("Saved feature summary CSV to", run_dir / "ig_feature_summary.csv")
    print("Saved raw attributions (npy) to", run_dir / "ig_clean.npy and ig_poison.npy")

if __name__ == "__main__":
    main()
