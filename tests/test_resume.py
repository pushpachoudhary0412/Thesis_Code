import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from mimiciv_backdoor_study.train_pipeline import (
    set_seed,
    build_dataset,
    build_model,
    train,
)
from torch import nn

def test_train_resume(tmp_path):
    # Prepare dataset and dataloader (dev subset)
    set_seed(0)
    base = Path("mimiciv_backdoor_study") / "data"
    splits = base / "splits" / "splits.json"

    ds_train, _ = build_dataset(base, splits, split="train", trigger=None, poison_rate=0.0, seed=0)
    train_loader = DataLoader(ds_train, batch_size=16, shuffle=True)

    input_dim = len(ds_train.feature_cols)
    device = "cpu"

    # Initial short training (1 epoch) that writes epoch1.pt (full checkpoint)
    model = build_model("mlp", input_dim, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = tmp_path / "run_resume"
    res1 = train(model, train_loader, None, opt, loss_fn, device, epochs=1, run_dir=run_dir)
    assert "last_checkpoint" in res1
    assert Path(res1["last_checkpoint"]).exists()

    # Start a fresh process (new model + optimizer) and resume from the saved checkpoint to reach epoch 2
    model2 = build_model("mlp", input_dim, device)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    res2 = train(model2, train_loader, None, opt2, loss_fn, device, epochs=2, run_dir=run_dir, resume_checkpoint=res1["last_checkpoint"])
    assert "last_checkpoint" in res2
    assert Path(res2["last_checkpoint"]).exists()
    # verify final checkpoint corresponds to epoch2
    assert str(res2["last_checkpoint"]).endswith("epoch2.pt")
