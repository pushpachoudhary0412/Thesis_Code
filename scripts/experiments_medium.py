"""
Medium-scale experiment driver

Runs experiments across models (MLP, LSTM, TCN), multiple seeds and poison rates,
collects IG attribution statistics and basic accuracy metrics.

Usage (example):
  python scripts/experiments_medium.py --run_dir runs/experiments --seeds 0 1 2 3 4 \
    --poison_rates 0.01 0.05 0.1 --n_samples 128 --epochs 10

Notes:
 - This is intentionally simple: runs sequentially on the available device (CPU or GPU).
 - Results are saved to CSV files under run_dir.
"""
import argparse
from pathlib import Path
import csv
import time
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except Exception:
    raise RuntimeError("torch is required to run experiments")

from mimiciv_backdoor_study.data_utils.dataset import TabularDataset, TriggeredDataset
from mimiciv_backdoor_study import explainability
from mimiciv_backdoor_study.models.mlp import MLP
from mimiciv_backdoor_study.models.lstm import LSTMModel
from mimiciv_backdoor_study.models.tcn import TemporalCNN
from mimiciv_backdoor_study.models.tabtransformer import SimpleTabTransformer
from mimiciv_backdoor_study.metrics import classification_metrics, expected_calibration_error, backdoor_metrics

def instantiate_model(name: str, input_dim: int, device: str):
    if name == "mlp":
        model = MLP(input_dim=input_dim, hidden_dims=(128, 64), n_classes=2, dropout=0.1)
    elif name == "lstm":
        model = LSTMModel(input_dim=input_dim, emb_dim=32, hidden_dim=64, n_classes=2)
    elif name == "tcn":
        model = TemporalCNN(input_dim=input_dim, channels=(32, 64), n_classes=2)
    elif name == "tabtransformer":
        # Simple TabTransformer-like model for tabular inputs
        model = SimpleTabTransformer(input_dim=input_dim, embed_dim=32, n_heads=4, n_layers=2, n_classes=2, dropout=0.1)
    else:
        raise ValueError(f"Unknown model {name}")
    return model.to(device)

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / n if n > 0 else 0.0

def eval_accuracy(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            n += x.size(0)
    return correct / n if n > 0 else 0.0

def compute_ig(model, inputs_np, steps, device):
    return explainability.EXPLAINERS.integrated_gradients(model, inputs_np, steps=steps, device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--poison_rates", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--models", nargs="+", default=["mlp","lstm","tcn"])
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # data paths (main MIMIC dataset)
    base = Path(__file__).resolve().parents[1] / "mimiciv_backdoor_study" / "data"
    parquet_path = base / "main.parquet"
    splits_json = base / "splits_main.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Main parquet not found at {parquet_path}; ensure MIMIC data is available")

    summary_path = run_dir / "experiment_summary.csv"
    # write CSV header
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp","model","seed","poison_rate","epochs","n_samples","device","train_loss","acc_clean","acc_poison","auroc_clean","auroc_poison","precision_clean","recall_clean","f1_clean","ece_clean","ece_poison","ASR","confidence_shift","mean_abs_trigger","mean_abs_others","run_time_s"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    for model_name in args.models:
        for seed in args.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            for poison_rate in args.poison_rates:
                t0 = time.time()
                # prepare data
                ds_train = TabularDataset(parquet_path, splits_json, split="train")
                ds_test_clean = TabularDataset(parquet_path, splits_json, split="test")
                ds_test_poisoned = TriggeredDataset(ds_test_clean, trigger_type="rare_value", poison_rate=poison_rate, seed=seed)

                train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
                # instantiate model
                input_dim = len(ds_train.feature_cols)
                model = instantiate_model(model_name, input_dim, device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = nn.CrossEntropyLoss()

                # training loop
                train_loss = 0.0
                for ep in range(args.epochs):
                    train_loss = train_epoch(model, train_loader, opt, loss_fn, device)

                # evaluate accuracies (and collect predictions/probabilities for metrics)
                acc_clean = eval_accuracy(model, ds_test_clean, device, batch_size=args.batch_size)
                acc_poison = eval_accuracy(model, ds_test_poisoned, device, batch_size=args.batch_size)

                # helper: collect true labels, preds and probs for a dataset
                def _get_preds_probs(model, dataset, device, batch_size):
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                    model.eval()
                    preds_list = []
                    probs_list = []
                    trues_list = []
                    with torch.no_grad():
                        for batch in loader:
                            x = batch["x"].to(device)
                            y = batch["y"].to(device)
                            out = model(x)
                            prob = torch.softmax(out, dim=1).cpu().numpy()
                            pred = out.argmax(dim=1).cpu().numpy()
                            preds_list.append(pred)
                            probs_list.append(prob)
                            trues_list.append(y.cpu().numpy())
                    if len(preds_list) > 0:
                        preds = np.concatenate(preds_list, axis=0)
                        probs = np.vstack(probs_list)
                        trues = np.concatenate(trues_list, axis=0)
                    else:
                        preds = np.array([], dtype=int)
                        probs = np.array([], dtype=float)
                        trues = np.array([], dtype=int)
                    return trues, preds, probs

                y_clean_true, y_clean_pred, y_clean_prob = _get_preds_probs(model, ds_test_clean, device, batch_size=args.batch_size)
                y_poison_true, y_poison_pred, y_poison_prob = _get_preds_probs(model, ds_test_poisoned, device, batch_size=args.batch_size)

                # compute classification metrics + calibration
                cm_clean = classification_metrics(y_clean_true, y_clean_pred, y_clean_prob)
                cm_poison = classification_metrics(y_poison_true, y_poison_pred, y_poison_prob)
                ece_clean = expected_calibration_error(y_clean_true, y_clean_prob)
                ece_poison = expected_calibration_error(y_poison_true, y_poison_prob)

                # backdoor-specific metrics
                bd = backdoor_metrics(y_clean_pred, y_clean_prob, y_poison_pred, y_poison_prob, target_label=None)

                # gather samples for IG (first n_samples)
                n = min(args.n_samples, len(ds_test_clean))
                clean_inputs = np.vstack([ds_test_clean[i]["x"].numpy() for i in range(n)])
                poison_inputs = np.vstack([ds_test_poisoned[i]["x"].numpy() for i in range(n)])

                ig_clean = compute_ig(model, clean_inputs, steps=args.steps, device=device)
                ig_poison = compute_ig(model, poison_inputs, steps=args.steps, device=device)

                # metric: mean abs attributions on trigger feature (index 0) vs others
                mean_abs_trigger = (np.mean(np.abs(ig_poison[:, 0])) + np.mean(np.abs(ig_clean[:, 0]))) / 2.0
                mean_abs_others = (np.mean(np.abs(ig_poison[:, 1:])) + np.mean(np.abs(ig_clean[:, 1:]))) / 2.0

                run_time = time.time() - t0

                # save per-run artifacts
                run_name = f"{model_name}_seed{seed}_pr{poison_rate:.3f}"
                np.save(run_dir / f"ig_{run_name}_clean.npy", ig_clean)
                np.save(run_dir / f"ig_{run_name}_poison.npy", ig_poison)

                # append summary row (include extended metrics)
                with open(summary_path, "a", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                     model_name, seed, poison_rate, args.epochs, n, device,
                                     f"{train_loss:.6f}", f"{acc_clean:.4f}", f"{acc_poison:.4f}",
                                     f"{cm_clean['auroc']:.6f}", f"{cm_poison['auroc']:.6f}",
                                     f"{cm_clean['precision']:.6f}", f"{cm_clean['recall']:.6f}", f"{cm_clean['f1']:.6f}",
                                     f"{ece_clean:.6f}", f"{ece_poison:.6f}",
                                     f"{bd['ASR']:.6f}", f"{bd['confidence_shift']:.6f}",
                                     f"{mean_abs_trigger:.8f}", f"{mean_abs_others:.8f}", f"{run_time:.2f}"])

                print(f"Completed {run_name} in {run_time:.1f}s: acc_clean={acc_clean:.3f} acc_poison={acc_poison:.3f} mean_abs_trigger={mean_abs_trigger:.6f}")

    print("All experiments complete. Summary saved to", summary_path)

if __name__ == "__main__":
    main()
