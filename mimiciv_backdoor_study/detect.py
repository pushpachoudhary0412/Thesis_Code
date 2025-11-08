#!/usr/bin/env python3
"""
Simple detection stub using attribution magnitudes (Captum) as a heuristic.

Writes run_dir/results_detect.json containing:
 - scores: per-sample anomaly score (mean absolute attribution)
 - flagged: indices flagged as suspicious (score > threshold)
 - threshold: chosen threshold (quantile)

Usage:
 PYTHONPATH=$(pwd) python mimiciv_backdoor_study/detect.py --run_dir mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42
"""
import argparse
import json
from pathlib import Path
import importlib

# Dynamically import numpy to avoid static analyzers (e.g. Pylance) raising
# "Import 'numpy' could not be resolved" in environments where numpy isn't installed.
# At runtime, if numpy is available we load it; otherwise fall back to None.
if importlib.util.find_spec("numpy") is not None:
    np = importlib.import_module("numpy")
    NUMPY_AVAILABLE = True
else:
    np = None
    NUMPY_AVAILABLE = False

import importlib
# Dynamically import torch and DataLoader to avoid static analyzers (e.g. Pylance)
# raising "Import 'torch.utils.data' could not be resolved" when torch is not available.
if importlib.util.find_spec("torch") is not None:
    torch = importlib.import_module("torch")
    try:
        DataLoader = getattr(importlib.import_module("torch.utils.data"), "DataLoader")
    except Exception:
        DataLoader = None
else:
    torch = None
    DataLoader = None

# Small helpers to avoid a hard dependency on numpy at runtime.
# If numpy is available we use it; otherwise fall back to torch-based implementations.
if NUMPY_AVAILABLE:
    def to_numpy(seq):
        return np.array(seq)
    def quantile(arr, q):
        return float(np.quantile(arr, q))
else:
    def to_numpy(seq):
        # create a torch tensor and convert to python list then to numpy via the standard library-free path
        return torch.tensor(seq, dtype=torch.float32).cpu().numpy()
    def quantile(arr, q):
        t = torch.tensor(arr, dtype=torch.float32)
        return float(torch.quantile(t, q).item())

from mimiciv_backdoor_study.data_utils.dataset import TabularDataset, TriggeredDataset, set_seed

# Dynamically import Captum (similar to numpy/torch) to avoid static analysis errors
# (e.g. Pylance complaining "Import 'captum.attr' could not be resolved")
if importlib.util.find_spec("captum.attr") is not None:
    try:
        Saliency = getattr(importlib.import_module("captum.attr"), "Saliency")
        CAPTUM_AVAILABLE = True
    except Exception:
        Saliency = None
        CAPTUM_AVAILABLE = False
else:
    Saliency = None
    CAPTUM_AVAILABLE = False

def compute_saliency_scores(model, loader, device):
    model.eval()
    scores = []
    if not CAPTUM_AVAILABLE:
        # fallback: use input L2 norm as a proxy score
        with torch.no_grad():
            for b in loader:
                x = b["x"].to(device)
                # avoid calling tensor.numpy() directly to reduce reliance on numpy at this call site;
                # use .tolist() which returns Python floats
                per = x.abs().mean(dim=1).cpu().tolist()
                scores.extend(per)
        return to_numpy(scores)

    saliency = Saliency(model)
    for b in loader:
        x = b["x"].to(device)  # (B, F)
        x.requires_grad_()
        # For classification, compute gradient w.r.t. target logit (predicted class)
        logits = model(x)
        preds = logits.argmax(dim=1)
        batch_scores = []
        for i in range(x.shape[0]):
            grad = saliency.attribute(x[i].unsqueeze(0), target=preds[i].item())
            score = grad.abs().mean().item()
            batch_scores.append(score)
        scores.extend(batch_scores)
    return to_numpy(scores)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, default=Path("mimiciv_backdoor_study/runs/mlp/none/0.0/seed_42"))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--threshold_quantile", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", type=str, default="saliency", help="detection method: saliency|activation_clustering|spectral")
    p.add_argument("--top_k", type=int, default=1, help="top-k vectors for spectral detector")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    dev_parquet = Path("mimiciv_backdoor_study/data/dev/dev.parquet")
    splits_json = Path("mimiciv_backdoor_study/data/splits/splits.json")

    # Load checkpoint (supports either a saved state_dict or a full model object)
    state = torch.load(ckpt, map_location=device)

    # Reconstruct model if a state_dict was saved (in that case infer input dim from a sample)
    from mimiciv_backdoor_study.models.mlp import MLP

    sample_for_shape = TabularDataset(dev_parquet, splits_json, split="train")[0]
    input_dim = sample_for_shape["x"].shape[0]

    if isinstance(state, dict):
        model = MLP(input_dim=input_dim)
        model.load_state_dict(state)
    else:
        model = state

    model = model.to(device)

    test_ds = TabularDataset(dev_parquet, splits_json, split="test")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Select detector
    if args.method == "saliency":
        scores = compute_saliency_scores(model, loader, device)
    elif args.method == "activation_clustering":
        from mimiciv_backdoor_study.detectors.activation_clustering import detect as ac_detect
        scores = ac_detect(model, loader, device)
    elif args.method == "spectral":
        from mimiciv_backdoor_study.detectors.spectral_signature import detect as ss_detect
        scores = ss_detect(model, loader, device, top_k=args.top_k)
    else:
        raise ValueError(f"unknown detection method: {args.method}")

    threshold = quantile(scores, args.threshold_quantile)
    # flagged as indices where score > threshold (numpy-free)
    flagged = [i for i, s in enumerate(scores) if s > threshold]

    out = {
        "scores": scores.tolist(),
        "threshold": threshold,
        "threshold_quantile": args.threshold_quantile,
        "flagged_indices": flagged,
        "num_flagged": len(flagged),
        "run_dir": str(run_dir),
        "captum_available": CAPTUM_AVAILABLE,
    }

    out_path = run_dir / "results_detect.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote detection results to {out_path} (flagged {len(flagged)} samples)")

if __name__ == "__main__":
    main()
