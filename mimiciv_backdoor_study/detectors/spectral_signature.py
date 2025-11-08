"""
Spectral signature detector (baseline).

API:
  def detect(model, loader, device, top_k=1) -> np.ndarray(scores)

This implementation:
 - collects per-sample representations (logits)
 - centers the representations and computes top-k singular vectors (SVD)
 - scores samples by their squared projection magnitude onto the top-k subspace
 - normalises scores to [0,1]
"""
from typing import Iterable

# numpy may not be available to the editor's type checker in some environments; silence Pylance.
import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]

def detect(model: torch.nn.Module, loader: Iterable, device: str, top_k: int = 1) -> np.ndarray:
    model.eval()
    reps = []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            logits = model(x)  # (B, C) - use logits as representation
            reps.append(logits.detach().cpu().numpy())
    if len(reps) == 0:
        return np.array([])

    reps = np.vstack(reps)  # (N, D)
    if reps.shape[0] < 2 or np.allclose(reps.std(axis=0), 0):
        return np.zeros(reps.shape[0])

    # Center
    mean = reps.mean(axis=0, keepdims=True)
    X = reps - mean  # (N, D)

    # Compute top-k left singular vectors via SVD on X (N x D)
    # We only need top_k singular vectors of the feature space (right singular vectors)
    # Use economy SVD
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        # Vt shape (D, D), top_k right-singular vectors are Vt[:top_k]
        top_vecs = Vt[:top_k]  # (top_k, D)
    except Exception:
        # Fallback to PCA via eig on covariance
        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # sort descending
        idx = np.argsort(eigvals)[::-1]
        top_vecs = eigvecs[:, idx[:top_k]].T

    # Project each centered sample onto top-k subspace and compute squared magnitude
    projections = X @ top_vecs.T  # (N, top_k)
    sq_mag = np.sum(projections ** 2, axis=1)  # (N,)

    # Normalise to [0,1]
    if sq_mag.max() > sq_mag.min():
        scores = (sq_mag - sq_mag.min()) / (sq_mag.max() - sq_mag.min())
    else:
        scores = sq_mag

    return scores
