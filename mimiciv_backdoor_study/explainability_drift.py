"""
Explainability drift utilities.

Provides functions to compare attributions (IG, SHAP) and attention maps between
clean and poisoned model runs.

Key functions:
 - attribution_distance(a, b, metric="l2"|"cosine") -> scalar or per-sample array
 - feature_rank_change(a, b, topk=None) -> dict with spearman_r, mean_abs_rank_shift, topk_overlap
 - trigger_attribution_ratio(attrib, trigger_mask) -> scalar (or per-sample mean)
 - attention_shift(attn_clean, attn_poison, metric="l1"|"kl") -> scalar or per-head/per-feature summary

Notes:
 - Inputs may be 1D (n_features) or 2D (n_samples, n_features). Functions handle both.
 - Uses numpy and scipy.stats where appropriate.
"""
from typing import Optional, Dict, Tuple
import numpy as np
from scipy.stats import spearmanr
from scipy.special import rel_entr
from scipy.stats import entropy as _entropy

__all__ = [
    "attribution_distance",
    "feature_rank_change",
    "trigger_attribution_ratio",
    "attention_shift",
    "normalize_abs",
    "rank_features",
]

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a

def normalize_abs(attrib: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """
    Return normalized absolute attributions so rows sum to 1.
    - attrib: shape (n_samples, n_features) or (n_features,)
    - axis: axis along which to normalize (default 1 -> per-sample)
    """
    a = _ensure_2d(attrib)
    a = np.abs(a)
    denom = a.sum(axis=axis, keepdims=True)
    denom = denom + eps
    return (a / denom).astype(float)

def attribution_distance(a: np.ndarray, b: np.ndarray, metric: str = "l2") -> np.ndarray:
    """
    Compute distance between attributions a and b.
    - Supports metric 'l2' (Euclidean) or 'cosine' (1 - cosine_similarity).
    - If inputs are 2D (n_samples, n_features), returns per-sample distances (shape (n_samples,))
      If 1D, returns scalar array with shape (1,).
    """
    a2 = _ensure_2d(a)
    b2 = _ensure_2d(b)
    if a2.shape != b2.shape:
        raise ValueError(f"Shapes must match: {a2.shape} vs {b2.shape}")
    if metric == "l2":
        diffs = a2 - b2
        d = np.linalg.norm(diffs, axis=1)
        return d
    elif metric == "cosine":
        # cosine distance = 1 - cosine_similarity
        num = (a2 * b2).sum(axis=1)
        a_norm = np.linalg.norm(a2, axis=1)
        b_norm = np.linalg.norm(b2, axis=1)
        denom = a_norm * b_norm
        denom = np.where(denom == 0.0, 1e-12, denom)
        cos = num / denom
        return 1.0 - cos
    else:
        raise ValueError("Unsupported metric: choose 'l2' or 'cosine'")

def rank_features(attrib: np.ndarray, descending: bool = True) -> np.ndarray:
    """
    Return integer ranks for features.
    - attrib: (n_samples, n_features) or (n_features,)
    - Returns ranks with 1 = highest attribution.
    """
    a2 = _ensure_2d(attrib)
    # rank by absolute value by default
    vals = np.abs(a2)
    # argsort descending
    order = np.argsort(-vals, axis=1)
    n_samples, n_features = vals.shape
    ranks = np.zeros_like(order, dtype=int)
    # convert order to ranks
    for i in range(n_samples):
        ranks[i, order[i]] = np.arange(1, n_features + 1)
    if attrib.ndim == 1:
        return ranks[0]
    return ranks

def feature_rank_change(a: np.ndarray, b: np.ndarray, topk: Optional[int] = None) -> Dict[str, float]:
    """
    Compute rank-change statistics between attribution vectors a and b.
    Returns:
      - spearman_r: Spearman correlation between mean attributions (scalar)
      - mean_abs_rank_shift: mean absolute change in rank across features
      - topk_overlap: Jaccard overlap of top-k features (if topk provided)
    Notes:
      - If inputs are 2D, computes mean attribution per-feature across samples before ranking.
    """
    a2 = _ensure_2d(a)
    b2 = _ensure_2d(b)
    if a2.shape[1] != b2.shape[1]:
        raise ValueError("Feature dimension mismatch")

    # compute mean per-feature
    a_mean = a2.mean(axis=0)
    b_mean = b2.mean(axis=0)

    # spearman correlation
    try:
        spearman_r, _ = spearmanr(a_mean, b_mean)
        if np.isnan(spearman_r):
            spearman_r = float("nan")
    except Exception:
        spearman_r = float("nan")

    # ranks
    ranks_a = rank_features(a_mean)
    ranks_b = rank_features(b_mean)
    # mean absolute rank shift
    mean_abs_rank_shift = float(np.mean(np.abs(ranks_a - ranks_b)))

    res: Dict[str, float] = {"spearman_r": float(spearman_r), "mean_abs_rank_shift": mean_abs_rank_shift}

    if topk is not None and topk > 0:
        top_a = set(np.argsort(-np.abs(a_mean))[:topk].tolist())
        top_b = set(np.argsort(-np.abs(b_mean))[:topk].tolist())
        inter = top_a.intersection(top_b)
        union = top_a.union(top_b)
        jaccard = float(len(inter) / len(union)) if len(union) > 0 else 0.0
        res["topk_jaccard"] = jaccard
        res["topk_overlap_count"] = float(len(inter))
    return res

def trigger_attribution_ratio(attrib: np.ndarray, trigger_mask: np.ndarray) -> np.ndarray:
    """
    Compute Trigger Attribution Ratio (TAR) = sum_abs(attrib[trigger]) / sum_abs(all features)
    - attrib: (n_samples, n_features) or (n_features,)
    - trigger_mask: boolean mask of length n_features (or indices)
    Returns per-sample TAR (shape (n_samples,)) or scalar if single sample.
    """
    a2 = _ensure_2d(attrib)
    if isinstance(trigger_mask, (list, tuple, np.ndarray)):
        mask = np.array(trigger_mask, dtype=bool)
    else:
        raise ValueError("trigger_mask must be array-like of booleans or indices")
    if mask.ndim != 1:
        mask = mask.flatten()
    if mask.shape[0] != a2.shape[1]:
        raise ValueError("trigger_mask length must equal number of features")
    num = np.abs(a2[:, mask]).sum(axis=1)
    denom = np.abs(a2).sum(axis=1)
    denom = np.where(denom == 0.0, 1e-12, denom)
    tar = num / denom
    if attrib.ndim == 1:
        return tar[0]
    return tar

def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    # p, q are 1D distributions
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    # use scipy rel_entr or manual
    return float(_entropy(p, q))

def attention_shift(attn_clean: np.ndarray, attn_poison: np.ndarray, metric: str = "l1") -> Dict[str, float]:
    """
    Compute attention shift between clean and poison attention arrays.

    Expected shapes:
      - (n_heads, n_features)
      - (n_samples, n_heads, n_features)
      - (n_samples, n_features) (if already averaged across heads)

    Returns dictionary with:
      - mean_l1: mean L1 difference per-feature averaged across heads/samples
      - mean_l2: mean L2 difference
      - mean_kl: mean KL divergence between normalized attention vectors (per head/sample)
    """
    a = np.asarray(attn_clean)
    b = np.asarray(attn_poison)
    if a.shape != b.shape:
        # try to broadcast if one side lacks sample dim
        if a.ndim == 2 and b.ndim == 3 and a.shape == b.shape[1:]:
            a = np.expand_dims(a, 0)
        elif b.ndim == 2 and a.ndim == 3 and b.shape == a.shape[1:]:
            b = np.expand_dims(b, 0)
        else:
            raise ValueError(f"Attention shapes must match or be broadcastable: {a.shape} vs {b.shape}")

    # normalize along feature axis for KL
    if a.ndim == 3:
        # shape (n_samples, n_heads, n_features) -> flatten sample+head for per-vector metrics
        n_samples, n_heads, n_features = a.shape
        a_flat = a.reshape(-1, n_features)
        b_flat = b.reshape(-1, n_features)
    elif a.ndim == 2:
        a_flat = a.reshape(-1, a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-1])
    else:
        raise ValueError("Unsupported attention array dimensionality")

    l1 = np.mean(np.sum(np.abs(a_flat - b_flat), axis=1))
    l2 = np.mean(np.linalg.norm(a_flat - b_flat, axis=1))
    # KL per vector using normalized distributions
    kl_vals = []
    for i in range(a_flat.shape[0]):
        p = a_flat[i]
        q = b_flat[i]
        # ensure non-negative and normalize
        p_pos = np.clip(p, 0.0, None)
        q_pos = np.clip(q, 0.0, None)
        # if all zeros, skip
        if p_pos.sum() == 0 or q_pos.sum() == 0:
            kl_vals.append(0.0)
        else:
            kl_vals.append(_kl_divergence(p_pos, q_pos))
    mean_kl = float(np.mean(kl_vals)) if kl_vals else 0.0

    return {"mean_l1": float(l1), "mean_l2": float(l2), "mean_kl": mean_kl}
