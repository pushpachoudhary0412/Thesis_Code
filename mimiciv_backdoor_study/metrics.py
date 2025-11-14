"""
Evaluation metrics for classification and backdoor experiments.

Provides:
- classification_metrics(y_true, y_pred, y_prob)
- expected_calibration_error(y_true, y_prob, n_bins=15)
- backdoor_metrics(y_clean_preds, y_clean_probs, y_poison_preds, y_poison_probs, target_label=None)

Dependencies: numpy, sklearn (for AUROC / precision/recall/f1). If sklearn is not available,
the functions will raise an informative ImportError.
"""
from typing import Dict, Optional, Tuple
import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_fscore_support,
        accuracy_score,
    )
except Exception as e:
    raise ImportError(
        "scikit-learn is required for mimiciv_backdoor_study.metrics. "
        "Install it with `pip install scikit-learn`."
    ) from e


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute common classification metrics.

    Args:
      y_true: (N,) ground-truth labels (0/1)
      y_pred: (N,) predicted labels
      y_prob: (N, C) predicted probabilities (C=2 expected for binary)

    Returns:
      dict with keys: accuracy, auroc, precision, recall, f1
    """
    out = {}
    if y_true.size == 0:
        return {
            "accuracy": float("nan"),
            "auroc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    try:
        acc = float(accuracy_score(y_true, y_pred))
    except Exception:
        acc = float("nan")
    out["accuracy"] = acc

    # AUROC (binary expected)
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
            scores = y_prob[:, 1]
        else:
            # if only single score provided, use it
            scores = y_prob.ravel()
        auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = float("nan")
    out["auroc"] = auroc

    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        out["precision"] = float(precision)
        out["recall"] = float(recall)
        out["f1"] = float(f1)
    except Exception:
        out["precision"] = float("nan")
        out["recall"] = float("nan")
        out["f1"] = float("nan")

    return out


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary classification.

    ECE = sum_k (|acc(B_k) - conf(B_k)| * |B_k|/N)

    Args:
      y_true: (N,) true labels
      y_prob: (N, C) predicted probabilities
      n_bins: number of probability bins

    Returns:
      ECE float
    """
    if y_true.size == 0:
        return float("nan")

    # predicted confidence and predicted label
    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        confidences = np.max(y_prob, axis=1)
        preds = np.argmax(y_prob, axis=1)
    else:
        confidences = y_prob.ravel()
        preds = (confidences >= 0.5).astype(int)

    N = y_true.shape[0]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idxs = np.digitize(confidences, bins, right=True) - 1
    bin_idxs = np.clip(bin_idxs, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idxs == b
        if not np.any(mask):
            continue
        acc_in_bin = np.mean((y_true[mask] == preds[mask]).astype(float))
        conf_in_bin = np.mean(confidences[mask])
        ece += np.abs(acc_in_bin - conf_in_bin) * (mask.sum() / N)
    return float(ece)


def backdoor_metrics(
    y_clean_preds: np.ndarray,
    y_clean_probs: np.ndarray,
    y_poison_preds: np.ndarray,
    y_poison_probs: np.ndarray,
    target_label: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute backdoor-specific metrics.

    Args:
      y_clean_preds / y_clean_probs: predictions & probs on clean test set
      y_poison_preds / y_poison_probs: predictions & probs on poisoned test set
      target_label: if known, ASR is fraction predicted == target_label;
                    otherwise inferred as the most common predicted label on poisoned set.

    Returns:
      dict with keys: ASR, CA, PA, confidence_shift
    """
    out = {}
    # CA / PA: accuracies are expected to be computed elsewhere; here we compute them as agreement
    # If ground-truth is not provided here, CA/PA must be passed; this function will compute ASR and confidence shift.
    # We'll compute CA/PA as accuracy vs inferred clean labels where possible (not ideal), so leave them as NaN
    out["CA"] = float("nan")
    out["PA"] = float("nan")

    # Infer target label if not provided
    if target_label is None:
        if y_poison_preds.size == 0:
            inferred = None
        else:
            # Most frequent predicted label on poisoned set
            vals, counts = np.unique(y_poison_preds, return_counts=True)
            inferred = int(vals[np.argmax(counts)])
        target_label = inferred

    # ASR: fraction of poisoned samples predicted as target_label
    try:
        if target_label is None:
            asr = float("nan")
        else:
            asr = float(np.mean((y_poison_preds == target_label).astype(float)))
    except Exception:
        asr = float("nan")
    out["ASR"] = asr

    # Confidence shift: mean(confidence for target class on poison) - mean(confidence for predicted class on clean)
    try:
        # confidence of target class on poisoned examples
        if y_poison_probs.ndim == 2 and y_poison_probs.shape[1] >= 2 and target_label is not None:
            conf_poison_target = y_poison_probs[:, target_label]
        else:
            # fallback: max prob
            conf_poison_target = np.max(y_poison_probs, axis=1) if y_poison_probs.size else np.array([])

        # confidence on clean (max predicted prob)
        conf_clean = np.max(y_clean_probs, axis=1) if y_clean_probs.size else np.array([])

        if conf_poison_target.size == 0 or conf_clean.size == 0:
            conf_shift = float("nan")
        else:
            conf_shift = float(np.mean(conf_poison_target) - np.mean(conf_clean))
    except Exception:
        conf_shift = float("nan")
    out["confidence_shift"] = conf_shift

    return out
