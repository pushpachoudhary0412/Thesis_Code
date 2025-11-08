try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover - provide a light fallback for editors without pytest
    class _PyTestStub:
        @staticmethod
        def importorskip(modname: str):
            import importlib
            return importlib.import_module(modname)
    pytest = _PyTestStub()

torch = pytest.importorskip("torch")
DataLoader = torch.utils.data.DataLoader
TensorDataset = torch.utils.data.TensorDataset
from mimiciv_backdoor_study.detectors import activation_clustering as ac
from mimiciv_backdoor_study.detectors import spectral_signature as ss


class DummyModel(torch.nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def make_loader(n_samples=10, input_dim=5, batch_size=4):
    gen = torch.Generator()
    gen.manual_seed(0)
    t = torch.randn((n_samples, input_dim), generator=gen, dtype=torch.float32)
    # labels not used by detectors but provide something
    y = torch.zeros((n_samples,), dtype=torch.long)
    ds = TensorDataset(t, y)
    # adapt to mimiciv_backdoor_study expected batch format: b["x"]
    def collate(batch):
        xs, ys = zip(*batch)
        return {"x": torch.stack(xs), "y": torch.tensor(ys)}
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def test_activation_clustering_outputs_shape_and_range():
    loader = make_loader(n_samples=12, input_dim=6, batch_size=5)
    model = DummyModel(input_dim=6, n_classes=3)
    device = "cpu"
    scores = ac.detect(model, loader, device)
    scores_t = torch.as_tensor(scores)
    assert scores_t.shape[0] == 12
    assert torch.all((scores_t >= 0.0) & (scores_t <= 1.0))


def test_spectral_signature_outputs_shape_and_range():
    loader = make_loader(n_samples=8, input_dim=4, batch_size=3)
    model = DummyModel(input_dim=4, n_classes=2)
    device = "cpu"
    scores = ss.detect(model, loader, device, top_k=1)
    scores_t = torch.as_tensor(scores)
    assert scores_t.shape[0] == 8
    assert torch.all((scores_t >= 0.0) & (scores_t <= 1.0))


def test_detectors_handle_constant_reps():
    # Create a loader where model returns identical logits for all samples
    class ConstModel(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            # return constant logits
            return torch.ones((B, 2)) * 0.5

    loader = make_loader(n_samples=5, input_dim=3, batch_size=5)
    model = ConstModel()
    device = "cpu"
    scores_ac = ac.detect(model, loader, device)
    scores_ss = ss.detect(model, loader, device, top_k=1)
    # detectors should return zero (or constant) scores, not NaN
    scores_ac_t = torch.as_tensor(scores_ac)
    scores_ss_t = torch.as_tensor(scores_ss)
    assert torch.all(torch.isfinite(scores_ac_t))
    assert torch.all(torch.isfinite(scores_ss_t))
    assert scores_ac_t.shape[0] == 5
    assert scores_ss_t.shape[0] == 5
