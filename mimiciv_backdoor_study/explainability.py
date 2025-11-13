"""
Explainability utilities (MVP)

Provides:
 - integrated_gradients: lightweight PyTorch implementation (no Captum dependency)
 - shap_explain: wrapper that uses SHAP if installed, otherwise raises ImportError
 - explain: thin dispatcher to chosen method

Design goals:
 - Keep dependencies optional (SHAP/Captum may be unavailable).
 - Provide a small, tested IG implementation suitable for tabular models used in this repo.
"""
from typing import Callable, Optional, Tuple
import numpy as np
import importlib
from types import SimpleNamespace

# Try to import torch at runtime (project uses torch extensively)
try:
    import torch
    from torch import Tensor
    from torch.nn import Module
except Exception:  # pragma: no cover - runtime will need torch
    torch = None
    Tensor = object
    Module = object


def integrated_gradients(
    model: Module,
    inputs: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    steps: int = 50,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients for a batch of inputs.

    Args:
      model: a PyTorch model (callable) returning logits or a single-scalar output per example.
      inputs: numpy array (batch_size, n_features)
      baseline: numpy array (n_features,) or (batch_size, n_features). If None, uses zero baseline.
      steps: integration steps
      device: optional device string, e.g. "cpu" or "cuda"

    Returns:
      attributions: numpy array same shape as inputs
    """
    if torch is None:
        raise ImportError("torch is required for integrated_gradients")

    model_device = torch.device(device) if device is not None else next(model.parameters()).device if any(
        p is not None for p in getattr(model, "parameters", lambda: [])()
    ) else torch.device("cpu")

    model.eval()

    x = torch.tensor(inputs, dtype=torch.float32, device=model_device)
    if baseline is None:
        baseline = np.zeros_like(inputs)
    baseline_t = torch.tensor(baseline, dtype=torch.float32, device=model_device)

    # If baseline has shape (n_features,), expand to batch
    if baseline_t.ndim == 1:
        baseline_t = baseline_t.unsqueeze(0).expand_as(x)

    # scaled inputs: baseline + alpha * (input - baseline)
    scaled_inputs = [
        baseline_t + (float(k) / steps) * (x - baseline_t) for k in range(1, steps + 1)
    ]

    grads = []
    for scaled in scaled_inputs:
        scaled.requires_grad = True
        out = model(scaled)
        # Support scalar outputs or binary logits. Sum outputs to get a single scalar per batch.
        if isinstance(out, tuple):
            out = out[0]
        # If output is multi-dim (batch, classes), reduce to predicted class score
        if out.dim() == 2:
            # choose predicted class per example
            scores = out.gather(1, out.argmax(dim=1, keepdim=True)).squeeze(1)
        else:
            scores = out.squeeze(1) if out.dim() == 2 else out.squeeze()
        # Sum scores to compute combined gradient
        total = scores.sum()
        total.backward()
        grads.append(scaled.grad.detach().clone())
        model.zero_grad()

    # average gradients
    avg_grads = torch.stack(grads, dim=0).mean(dim=0)  # (batch, features)
    attributions = (x - baseline_t) * avg_grads  # elementwise
    return attributions.cpu().numpy()


def shap_explain(
    model: Callable,
    background: np.ndarray,
    inputs: np.ndarray,
    nsamples: int = 100,
) -> np.ndarray:
    """
    Run SHAP explanation if shap is installed.

    Args:
      model: function that maps numpy inputs to model outputs (batch -> batch)
      background: small background dataset (n_background, n_features)
      inputs: inputs to explain (batch, n_features)
      nsamples: sampling budget for KernelExplainer (approximate)

    Returns:
      shap_values: numpy array (batch, n_features)
    """
    shap = importlib.import_module("shap")  # may raise ImportError
    # shap expects a function that returns prediction for each row
    explainer = shap.KernelExplainer(lambda x: model(x).astype(float), background)
    shap_values = explainer.shap_values(inputs, nsamples=nsamples)
    # KernelExplainer returns list for multiclass; for binary/regression it returns array
    if isinstance(shap_values, list):
        # pick the explanation for the predicted class (or first class)
        shap_arr = np.asarray(shap_values[0])
    else:
        shap_arr = np.asarray(shap_values)
    return shap_arr


def explain(
    model: Callable,
    inputs: np.ndarray,
    method: str = "ig",
    **kwargs,
) -> np.ndarray:
    """
    Unified API: explain(model, inputs, method="ig"|"shap", ...)

    - For method="ig", model should be a torch.nn.Module and inputs a numpy array.
    - For method="shap", model should be a numpy-callable (batch->batch) and a 'background' kwarg must be provided.

    Returns:
      attributions numpy array (batch, n_features)
    """
    method = (method or "ig").lower()
    if method == "ig":
        return integrated_gradients(model=model, inputs=inputs, **kwargs)
    if method == "shap":
        if "background" not in kwargs:
            raise ValueError("shap explain requires a 'background' ndarray kwarg")
        return shap_explain(model=model, background=kwargs["background"], inputs=inputs, nsamples=kwargs.get("nsamples", 100))
    raise ValueError(f"Unknown explanation method: {method}")


# Simple Namespace for convenience when importing in tests
EXPLAINERS = SimpleNamespace(integrated_gradients=integrated_gradients, shap_explain=shap_explain, explain=explain)
