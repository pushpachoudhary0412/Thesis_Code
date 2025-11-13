import numpy as np
import pytest

try:
    import torch
    from torch import nn
except Exception:
    pytest.skip("torch not available", allow_module_level=True)

from mimiciv_backdoor_study import explainability


def test_integrated_gradients_smoke():
    n_features = 5
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(n_features, 1))
    for p in model.parameters():
        nn.init.normal_(p, mean=0.0, std=0.1)
    model.eval()

    inputs = np.random.randn(2, n_features).astype(np.float32)
    attributions = explainability.EXPLAINERS.integrated_gradients(model, inputs, steps=5)
    assert attributions.shape == inputs.shape
    assert np.isfinite(attributions).all()
