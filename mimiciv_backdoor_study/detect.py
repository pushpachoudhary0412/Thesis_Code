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
    # Some older checkpoints require unpickling globals (e.g. numpy reconstruct). Try a safe default
    # load first; on UnpicklingError / weights-only error, allowlist numpy's reconstruct and retry
    try:
        state = torch.load(ckpt, map_location=device)
    except Exception as e:
        msg = str(e)
        # Detect the specific torch warning/error about weights-only / unsafe globals
        if "Weights only load" in msg or "weights_only" in msg or "Unsupported global" in msg or "UnpicklingError" in msg:
            try:
                # ensure we have numpy available (dynamic import above may have set `np`)
                if np is None:
                    import importlib as _importlib
                    _np = _importlib.import_module("numpy")
                else:
                    _np = np
                # allowlist numpy's internal reconstruct function so torch.load can unpickle safely
                try:
                    torch.serialization.add_safe_globals([_np.core.multiarray._reconstruct])
                except Exception:
                    # older torch APIs expose safe_globals as context manager
                    try:
                        torch.serialization.safe_globals([_np.core.multiarray._reconstruct])
                    except Exception:
                        pass
            except Exception:
                # if we can't import/allowlist numpy, fall back to attempting a non-weights-only load
                pass
            # retry load with weights_only=False to allow full object unpickling (trusted local files)
            state = torch.load(ckpt, map_location=device, weights_only=False)
        else:
            # unknown error -> re-raise
            raise

    # Reconstruct model using the correct class inferred from run metadata.
    # Many runs save full checkpoints (with keys like "model_state", "optimizer_state", "epoch")
    # and different runs use different model classes (mlp, lstm, tcn, tabtransformer).
    meta_path = run_dir / "run_metadata.json"
    model_name = None
    if meta_path.exists():
        try:
            meta = json.load(open(meta_path))
            model_name = meta.get("model")
        except Exception:
            model_name = None

    # Prefer run_metadata feature list to avoid needing parquet reader (pyarrow) during detection.
    input_dim = None
    if meta_path.exists():
        try:
            meta2 = json.load(open(meta_path))
            train_meta = meta2.get("train_meta", {})
            feature_cols = train_meta.get("feature_cols")
            if feature_cols:
                input_dim = len(feature_cols)
        except Exception:
            input_dim = None
    if input_dim is None:
        sample_for_shape = TabularDataset(dev_parquet, splits_json, split="train")[0]
        input_dim = sample_for_shape["x"].shape[0]

    if model_name == "lstm":
        from mimiciv_backdoor_study.models.lstm import LSTMModel as ModelClass
    elif model_name == "tcn":
        from mimiciv_backdoor_study.models.tcn import TemporalCNN as ModelClass
    elif model_name == "tabtransformer":
        from mimiciv_backdoor_study.models.tabtransformer import SimpleTabTransformer as ModelClass
    else:
        # default to MLP for backwards compatibility
        from mimiciv_backdoor_study.models.mlp import MLP as ModelClass

    # Debug: write chosen model class and input_dim into run_dir for triage
    try:
        debug_info = {
            "model_name": model_name,
            "ModelClass": ModelClass.__name__ if "ModelClass" in locals() else None,
            "input_dim": input_dim,
            "meta_feature_cols_present": bool('feature_cols' in locals() and feature_cols),
        }
        (run_dir / "detect_debug.json").write_text(json.dumps(debug_info))
    except Exception:
        # best-effort logging only
        pass

    model = ModelClass(input_dim=input_dim)

    # Normalize checkpoint formats and load weights when state is a dict.
    if isinstance(state, dict):
        # prefer explicit inner keys if present — support common checkpoint wrappers
        # Try a list of known candidate keys that may contain the model parameters.
        preferred_keys = [
            "model_state",
            "model_state_dict",
            "state_dict",
            "model",
            "module",
            "net",
            "network",
            "state",
        ]
        sd = None
        for k in preferred_keys:
            if k in state:
                sd = state[k]
                break
        # fallback to the checkpoint itself if no candidate key matched
        if sd is None:
            sd = state

        # Some checkpoints wrap the actual parameter dict in another mapping; try to find the parameter dict.
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        def strip_module_prefix(sd_in):
            # remove common DataParallel / DistributedDataParallel prefix "module."
            try:
                return { (k.replace("module.", "") if isinstance(k, str) else k): v for k, v in sd_in.items() }
            except Exception:
                return sd_in

        def _looks_like_state_dict(d):
            # heuristics: mapping with string keys containing dots or tensor-like values
            if not isinstance(d, dict):
                return False
            if not d:
                return False
            dot_keys = sum(1 for k in d.keys() if isinstance(k, str) and "." in k)
            tensor_vals = sum(
                1
                for v in d.values()
                if hasattr(v, "dim") or hasattr(v, "shape") or hasattr(v, "dtype")
            )
            return dot_keys > 0 or tensor_vals > 0

        def _find_candidate_state_dict(obj, max_depth=4):
            # recursively search for the largest dict that looks like a state_dict
            candidates = []
            def walk(o, depth=0):
                if depth > max_depth:
                    return
                if isinstance(o, dict):
                    if _looks_like_state_dict(o):
                        candidates.append(o)
                    for v in o.values():
                        walk(v, depth + 1)
                elif isinstance(o, (list, tuple)):
                    for v in o:
                        walk(v, depth + 1)
            walk(obj, 0)
            # prefer the candidate with most keys
            if not candidates:
                return None
            return max(candidates, key=lambda x: len(x.keys()))

        # Try a series of load attempts with sensible fallbacks:
        load_error = None
        tried_candidates = []

        def _try_load(sd_candidate):
            try:
                model.load_state_dict(sd_candidate)
                return True, None
            except Exception as e:
                return False, e

        # 1) direct sd
        if isinstance(sd, dict):
            success, err = _try_load(sd)
            tried_candidates.append(("direct", sd))
            if success:
                load_error = None
            else:
                load_error = err

            # 2) try stripping common "module." prefix if present
            if load_error is not None:
                stripped = strip_module_prefix(sd)
                if stripped is not sd:
                    success2, err2 = _try_load(stripped)
                    tried_candidates.append(("stripped_module", stripped))
                    if success2:
                        load_error = None
                    else:
                        load_error = err2

        # 3) if still failing, try to find a nested candidate state_dict anywhere in the checkpoint
        if load_error is not None:
            candidate = _find_candidate_state_dict(sd)
            if candidate is not None:
                tried_candidates.append(("nested_candidate", candidate))
                try:
                    model.load_state_dict(candidate)
                    load_error = None
                except Exception as e_nested:
                    # try stripping module prefix on the nested candidate too
                    try:
                        model.load_state_dict(strip_module_prefix(candidate))
                        load_error = None
                    except Exception as e_nested2:
                        raise RuntimeError(
                            f"Failed to load nested candidate state_dict into {model.__class__.__name__}: "
                            f"first={load_error}; nested1={e_nested}; nested2={e_nested2}"
                        )

        if load_error is not None:
            # Before failing hard, attempt a last-resort non-strict load which can succeed
            # when keys mismatch due to wrapper prefixes or missing optimizer keys.
            tried_candidates.append(("last_resort_non_strict", None))
            try:
                # try non-strict on the most recent sd/candidate we examined (if any)
                last_sd = None
                if tried_candidates:
                    # take the last non-name entry that was a mapping
                    for name, cand in reversed(tried_candidates):
                        if isinstance(cand, dict):
                            last_sd = cand
                            break
                if last_sd is not None:
                    try:
                        model.load_state_dict(last_sd, strict=False)
                        load_error = None
                    except Exception as e_ns:
                        # also try stripping module prefix then non-strict
                        try:
                            model.load_state_dict(strip_module_prefix(last_sd), strict=False)
                            load_error = None
                        except Exception as e_ns2:
                            load_error = e_ns2
                # final fallback: attempt a non-strict load from the original sd if available
                else:
                    try:
                        model.load_state_dict(sd, strict=False)
                        load_error = None
                    except Exception as e_ns3:
                        try:
                            model.load_state_dict(strip_module_prefix(sd), strict=False)
                            load_error = None
                        except Exception as e_ns4:
                            load_error = e_ns4
            except Exception:
                # ignore and fall through to raising the original informative error
                pass

        if load_error is not None:
            # No viable load strategy succeeded — surface an informative error including which candidates were tried.
            raise RuntimeError(
                f"Failed to load state_dict into {model.__class__.__name__}: {load_error}. Tried candidates: {[name for name, _ in tried_candidates]}"
            )
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
