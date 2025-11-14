import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from mimiciv_backdoor_study.train_pipeline import (
    set_seed,
    build_dataset,
    build_model,
    train,
    evaluate,
)
from torch import nn
import numpy as np

def test_train_pipeline_clean_smoke(tmp_path):
    set_seed(0)
    base = Path("mimiciv_backdoor_study") / "data"
    splits = base / "splits" / "splits.json"

    # small train dataset (uses full train split but that's ok for smoke)
    ds_train, _ = build_dataset(base, splits, split="train", trigger=None, poison_rate=0.0, seed=0)
    train_loader = DataLoader(ds_train, batch_size=16, shuffle=True)

    input_dim = len(ds_train.feature_cols)
    device = "cpu"
    model = build_model("mlp", input_dim, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = tmp_path / "run_clean"
    res = train(model, train_loader, None, opt, loss_fn, device, epochs=1, run_dir=run_dir)
    assert "last_checkpoint" in res
    assert Path(res["last_checkpoint"]).exists()

    metrics = evaluate(model, res["last_checkpoint"], base, splits, device, batch_size=64, trigger=None, poison_rate=0.0, seed=0)
    # basic sanity checks
    assert "acc_clean" in metrics
    assert isinstance(metrics["acc_clean"], float) or np.isnan(metrics["acc_clean"])

def test_train_pipeline_poisoned_smoke(tmp_path):
    set_seed(1)
    base = Path("mimiciv_backdoor_study") / "data"
    splits = base / "splits" / "splits.json"

    ds_train, _ = build_dataset(base, splits, split="train", trigger="rare_value", poison_rate=0.05, seed=1)
    train_loader = DataLoader(ds_train, batch_size=16, shuffle=True)

    input_dim = len(ds_train.feature_cols)
    device = "cpu"
    model = build_model("mlp", input_dim, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = tmp_path / "run_poison"
    res = train(model, train_loader, None, opt, loss_fn, device, epochs=1, run_dir=run_dir)
    assert "last_checkpoint" in res
    assert Path(res["last_checkpoint"]).exists()

    metrics = evaluate(model, res["last_checkpoint"], base, splits, device, batch_size=64, trigger="rare_value", poison_rate=0.05, seed=1)
    assert "ASR" in metrics
    assert isinstance(metrics["ASR"], float) or np.isnan(metrics["ASR"])
