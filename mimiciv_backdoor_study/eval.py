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

import sys
# Ensure project root is on PYTHONPATH when running this script directly.
# Insert the parent of the package directory so `import mimiciv_backdoor_study` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # type: ignore[import]
import numpy as np  # type: ignore[import]
try:
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False
    def accuracy_score(y_true, y_pred):
        # pure-Python fallback (works on lists)
        y_true = list(y_true)
        y_pred = list(y_pred)
        if len(y_true) == 0:
            return 0.0
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return float(correct) / len(y_true)
    def roc_auc_score(y_true, y_score):
        raise RuntimeError("scikit-learn not available; cannot compute ROC AUC")
    def average_precision_score(y_true, y_score):
        raise RuntimeError("scikit-learn not available; cannot compute Average Precision")

from mimiciv_backdoor_study.data_utils.dataset import TabularDataset, TriggeredDataset, set_seed
from mimiciv_backdoor_study.models.mlp import MLP
from mimiciv_backdoor_study.models.tabtransformer import SimpleTabTransformer
from mimiciv_backdoor_study.explainability import explain, EXPLAINERS


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
    # keep lists (probs, preds, targets) as plain Python lists
    metrics = {}
    # If targets contain only a single class (e.g., no positive examples), AUC/AP are undefined.
    # Guard against sklearn raising warnings/errors by skipping those computations in that case.
    if len(set(targets)) < 2:
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
    p.add_argument("--explain_method", type=str, default="none", help="explainability method: 'ig' or 'shap' or 'none'")
    p.add_argument("--explain_n_samples", type=int, default=10, help="number of samples to explain (small smoke test)")
    p.add_argument("--explain_background_size", type=int, default=50, help="background size for SHAP")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # dataset paths are relative to project layout used by train.py
    # When running from mimiciv_backdoor_study/, paths should be relative to current directory
    dev_parquet = Path("data/main.parquet")
    splits_json = Path("data/splits_main.json")

    # Check if paths exist; if not, try looking relative to mimiciv_backdoor_study directory
    if not dev_parquet.exists():
        alt_parquet = Path("mimiciv_backdoor_study/data/main.parquet")
        if alt_parquet.exists():
            dev_parquet = alt_parquet
        else:
            print(f"Warning: Neither {dev_parquet} nor {alt_parquet} found. Using synthetic fallback.")
    if not splits_json.exists():
        alt_splits = Path("mimiciv_backdoor_study/data/splits_main.json")
        if alt_splits.exists():
            splits_json = alt_splits
        else:
            print(f"Warning: Neither {splits_json} nor {alt_splits} found. Using synthetic split fallback.")

    # Reconstruct model architecture (train.py saves state_dict) and load checkpoint.
    sample_for_shape = TabularDataset(dev_parquet, splits_json, split="train")[0]
    input_dim = sample_for_shape["x"].shape[0]

    # Try to determine model type from run metadata or checkpoint
    model_type = "mlp"  # default
    if (run_dir / "run_metadata.json").exists():
        try:
            metadata = json.load(open(run_dir / "run_metadata.json"))
            model_type = metadata.get("model", "mlp")
        except:
            pass

    # Load checkpoint first to infer model type if needed
    state = torch.load(ckpt, map_location=device)

    # Check for model type in checkpoint metadata
    if isinstance(state, dict) and "config" in state:
        config = state["config"]
        model_type = config.get("model", model_type)

    # Instantiate the appropriate model
    if model_type == "tabtransformer":
        # Enable attention extraction for TabTransformer
        model = SimpleTabTransformer(input_dim=input_dim)
        model.extract_attention = True  # Enable attention saving
    else:
        # Default to MLP: hidden_dims=[512, 256, 128]
        model = MLP(input_dim=input_dim, hidden_dims=[512, 256, 128])

    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        # fallback if a full model object was saved
        model = state
    model = model.to(device)

    test_base = TabularDataset(dev_parquet, splits_json, split="test")

    # Use DataLoader via the already-imported `torch` to avoid static import resolution issues
    # (keep torch imported at top: `import torch`)

    # clean evaluation
    test_loader = torch.utils.data.DataLoader(test_base, batch_size=args.batch_size, shuffle=False)
    clean_metrics, _, _, _ = evaluate_model(model, test_loader, device)

    # poisoned evaluation (apply trigger to all or fraction)
    poisoned_ds = TriggeredDataset(test_base, trigger_type=args.trigger, poison_rate=args.poison_rate, seed=args.seed, target_label=args.target_label)
    poisoned_loader = torch.utils.data.DataLoader(poisoned_ds, batch_size=args.batch_size, shuffle=False)
    poisoned_metrics, poisoned_probs, poisoned_preds, poisoned_targets = evaluate_model(model, poisoned_loader, device)

    # Save trigger mask for TAR (Trigger Attribution Ratio) calculations
    try:
        trigger_mask = getattr(poisoned_ds, '_poisoned_mask', None)
        if trigger_mask is not None:
            trigger_mask_path = run_dir / "trigger_mask.npy"
            np.save(str(trigger_mask_path), trigger_mask)
            # Also write poisoned indices for auditability
            poisoned_indices = np.nonzero(trigger_mask)[0].tolist()
            with open(run_dir / "poisoned_indices.json", "w") as f:
                json.dump(poisoned_indices, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save trigger mask: {e}")

    # compute ASR: fraction of poisoned samples predicted as target_label
    # TriggeredDataset marks poisoned indices internally; we approximate ASR as fraction of samples predicted as target_label
    asr = float(np.mean(np.array(poisoned_preds) == args.target_label))

    # optional explainability (small smoke test) — compute and save clean + poisoned explanations
    explanations_path = None
    if args.explain_method and args.explain_method.lower() != "none":
        try:
            import numpy as _np
            # collect a small set of inputs from the clean test set
            n = min(args.explain_n_samples, len(test_base))
            clean_inputs = []
            for i in range(n):
                item = test_base[i]
                x_np = item["x"].numpy() if isinstance(item["x"], torch.Tensor) else _np.asarray(item["x"])
                clean_inputs.append(x_np)
            clean_inputs = _np.stack(clean_inputs, axis=0)

            # collect same number of inputs from the poisoned test set
            poisoned_samples = []
            for i in range(n):
                item = poisoned_ds[i]
                x_np = item["x"].numpy() if isinstance(item["x"], torch.Tensor) else _np.asarray(item["x"])
                poisoned_samples.append(x_np)
            poisoned_inputs = _np.stack(poisoned_samples, axis=0)

            # prepare SHAP background from the clean test set if needed
            background = None
            if args.explain_method.lower() == "shap":
                bg_n = min(args.explain_background_size, len(test_base))
                background = []
                for i in range(bg_n):
                    item = test_base[i]
                    background.append(item["x"].numpy() if isinstance(item["x"], torch.Tensor) else _np.asarray(item["x"]))
                background = _np.stack(background, axis=0)

                # numpy model wrapper returning probabilities for the positive class
                def _numpy_model(xarr):
                    with torch.no_grad():
                        t = torch.tensor(xarr, dtype=torch.float32, device=device)
                        logits = model(t)
                        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                        return probs

                explanations_clean = explain(_numpy_model, clean_inputs, method="shap", background=background, nsamples=50)
                explanations_poison = explain(_numpy_model, poisoned_inputs, method="shap", background=background, nsamples=50)
            else:
                explanations_clean = explain(model, clean_inputs, method="ig", steps=50, baseline=None)
                explanations_poison = explain(model, poisoned_inputs, method="ig", steps=50, baseline=None)

            # save both arrays for aggregator to compute explainability drift
            explanations_clean_path = run_dir / "explanations_clean.npy"
            explanations_poison_path = run_dir / "explanations_poison.npy"
            _np.save(str(explanations_clean_path), explanations_clean)
            _np.save(str(explanations_poison_path), explanations_poison)
            explanations_path = explanations_clean_path
        except Exception as e:
            print("Explainability failed:", e)

    # Save attention weights for TabTransformer models
    if model_type == "tabtransformer" and hasattr(model, 'saved_attention') and len(model.saved_attention) > 0:
        try:
            # Save attention weights from clean evaluation
            attn_clean_path = run_dir / "attn_clean.npy"
            np.save(str(attn_clean_path), model.saved_attention)
            print(f"Saved attention weights from clean evaluation to {attn_clean_path}")

            # Clear attention buffer
            model.saved_attention = []

            # Re-run poisoned evaluation to collect poisoned attention
            model.extract_attention = True
            _ = evaluate_model(model, poisoned_loader, device)[0]  # We only need the evaluation side effect for attention

            if len(model.saved_attention) > 0:
                attn_poison_path = run_dir / "attn_poison.npy"
                np.save(str(attn_poison_path), model.saved_attention)
                print(f"Saved attention weights from poisoned evaluation to {attn_poison_path}")
        except Exception as e:
            print(f"Warning: Failed to save attention weights: {e}")

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
