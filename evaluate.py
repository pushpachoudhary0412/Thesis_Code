#!/usr/bin/env python3
"""
Standalone evaluator that loads a checkpoint (optional) and runs evaluation on clean and poisoned test sets.

Usage:
  python evaluate.py --checkpoint runs/exp/epoch1.pt --model lstm --trigger rare_value --poison_rate 0.05
"""
import argparse
from pathlib import Path
import json
import torch

from mimiciv_backdoor_study.train_pipeline import build_model, evaluate, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--trigger", type=str, default=None)
    parser.add_argument("--poison_rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--splits_json", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[0] / "mimiciv_backdoor_study" / "data"
    data_dir = Path(args.data_dir) if args.data_dir else base
    splits_json = Path(args.splits_json) if args.splits_json else base / "splits" / "splits.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)

    # need input dim: load a small dataset to infer input dim
    # reuse build_model with input_dim derived from train dataset
    from mimiciv_backdoor_study.train_pipeline import build_dataset
    ds_train, _ = build_dataset(data_dir, splits_json, split="train", trigger=None, poison_rate=0.0, seed=args.seed)
    input_dim = len(ds_train.feature_cols)
    model = build_model(args.model, input_dim, device)

    metrics = evaluate(model, args.checkpoint, data_dir, splits_json, device,
                       batch_size=args.batch_size,
                       trigger=args.trigger,
                       poison_rate=args.poison_rate,
                       seed=args.seed)

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
