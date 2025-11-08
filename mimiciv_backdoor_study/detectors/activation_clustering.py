"""
Activation clustering detector (simple baseline).

API:
  def detect(model, loader, device) -> np.ndarray(scores)

This implementation uses model logits as representations (logits per sample),
clusters them with KMeans(k=2) and uses distance to nearest cluster center
as an anomaly score (higher = more anomalous).
"""

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # type: ignore
    import torch  # type: ignore

import numpy as np  # type: ignore


def detect(model: "torch.nn.Module", loader: Iterable, device: str) -> "np.ndarray":
    import importlib
    try:
        torch = importlib.import_module("torch")
    except Exception as e:
        raise ImportError(
            "PyTorch is required for activation_clustering.detect; "
            "install torch (e.g., pip install torch)"
        ) from e
    model.eval()
    reps = []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            logits = model(x)  # (B, C) - use logits as representation
            reps.append(logits.detach().cpu().numpy())
    if len(reps) == 0:
        return np.array([])

    reps = np.vstack(reps)  # (N, C)
    # If only a single unique row, return zeros
    if reps.shape[0] < 2 or np.allclose(reps.std(axis=0), 0):
        return np.zeros(reps.shape[0])

    # Fit KMeans with 2 clusters (clean vs poisoned assumption)
    try:
        import importlib
        sklearn_cluster = importlib.import_module("sklearn.cluster")
        KMeans = sklearn_cluster.KMeans
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for activation_clustering.detect; "
            "install scikit-learn (e.g., pip install scikit-learn)"
        ) from e
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(reps)
    centers = kmeans.cluster_centers_
    # compute distance to assigned center (Euclidean)
    dists = np.linalg.norm(reps - centers[kmeans.labels_], axis=1)
    # Normalize to [0,1]
    if dists.max() > 0:
        scores = (dists - dists.min()) / (dists.max() - dists.min())
    else:
        scores = dists
    return scores
