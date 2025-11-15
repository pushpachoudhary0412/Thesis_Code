"""
Aggregate experiment results and produce simple plots for the proposal.

Usage:
  python scripts/aggregate_experiment_results.py --run_dir runs/experiments --out_dir runs/experiments/summary

Outputs:
 - summary CSV (aggregated metrics)
 - simple PNGs: mean_abs_trigger_by_model_pr.png, acc_by_model_pr.png, asr_by_model_pr.png,
   confidence_shift_by_model_pr.png, ece_clean_by_model_pr.png, ece_poison_by_model_pr.png
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def _plot_with_std(agg_df, x_col, y_col_mean="mean", y_col_std="std", out_path=None, xlabel="", ylabel="", title=""):
    plt.figure(figsize=(6,4))
    for model in agg_df['model'].unique():
        sub = agg_df[agg_df['model'] == model].sort_values(x_col)
        x = sub[x_col].to_numpy()
        y = sub[y_col_mean].to_numpy()
        plt.plot(x, y, marker="o", label=str(model))
        if y_col_std in sub.columns:
            std = sub[y_col_std].to_numpy()
            plt.fill_between(x, y - std, y + std, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150)
    plt.close()

def aggregate(run_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all per-run experiment_summary.csv files recursively and merge them
    files = list(Path(run_dir).rglob("experiment_summary.csv"))
    if not files:
        raise FileNotFoundError(f"No experiment_summary.csv files found under {run_dir}. Run experiments first.")

    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            dfs.append(d)
        except Exception:
            # skip files that fail to read
            continue

    if not dfs:
        raise RuntimeError("No valid CSVs found to aggregate.")

    df = pd.concat(dfs, ignore_index=True)

    # ensure poison_rate is numeric for sorting/plots
    if "poison_rate" in df.columns:
        try:
            df["poison_rate"] = pd.to_numeric(df["poison_rate"], errors="coerce")
        except Exception:
            pass

    # save a merged cleaned copy
    df.to_csv(out_dir / "experiment_summary_raw.csv", index=False)

    # aggregate: mean and std of mean_abs_trigger per model x poison_rate
    if "mean_abs_trigger" in df.columns:
        agg = df.groupby(["model","poison_rate"]).mean_abs_trigger.agg(["mean","std","count"]).reset_index()
        agg.to_csv(out_dir / "mean_abs_trigger_by_model_pr.csv", index=False)
        _plot_with_std(agg, x_col="poison_rate", out_path=out_dir / "mean_abs_trigger_by_model_pr.png",
                       xlabel="Poison rate", ylabel="Mean abs IG attribution (trigger)",
                       title="Mean abs IG attribution on trigger feature by model")

    # plot accuracies per model/pr (mean)
    if {"acc_clean","acc_poison"}.issubset(df.columns):
        agg_acc = df.groupby(["model","poison_rate"])[["acc_clean","acc_poison"]].mean().reset_index()
        plt.figure(figsize=(6,4))
        for model in agg_acc['model'].unique():
            sub = agg_acc[agg_acc['model'] == model].sort_values("poison_rate")
            x = sub["poison_rate"].to_numpy()
            y = sub["acc_poison"].to_numpy()
            plt.plot(x, y, marker="o", label=str(model))
        plt.xlabel("Poison rate")
        plt.ylabel("Mean poisoned accuracy")
        plt.title("Poisoned accuracy by model")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "acc_poison_by_model_pr.png", dpi=150)
        plt.close()

    # Aggregate ASR
    if "ASR" in df.columns:
        agg_asr = df.groupby(["model","poison_rate"])["ASR"].agg(["mean","std","count"]).reset_index()
        agg_asr.to_csv(out_dir / "asr_by_model_pr.csv", index=False)
        _plot_with_std(agg_asr, x_col="poison_rate", out_path=out_dir / "asr_by_model_pr.png",
                       xlabel="Poison rate", ylabel="ASR (attack success rate)",
                       title="ASR by model and poison rate")

    # Aggregate confidence shift
    if "confidence_shift" in df.columns:
        agg_conf = df.groupby(["model","poison_rate"])["confidence_shift"].agg(["mean","std","count"]).reset_index()
        agg_conf.to_csv(out_dir / "confidence_shift_by_model_pr.csv", index=False)
        _plot_with_std(agg_conf, x_col="poison_rate", out_path=out_dir / "confidence_shift_by_model_pr.png",
                       xlabel="Poison rate", ylabel="Confidence shift",
                       title="Confidence shift (poison - clean) by model and poison rate")

    # Aggregate ECE (clean / poison)
    if "ece_clean" in df.columns:
        agg_ece_clean = df.groupby(["model","poison_rate"])["ece_clean"].agg(["mean","std","count"]).reset_index()
        agg_ece_clean.to_csv(out_dir / "ece_clean_by_model_pr.csv", index=False)
        _plot_with_std(agg_ece_clean, x_col="poison_rate", out_path=out_dir / "ece_clean_by_model_pr.png",
                       xlabel="Poison rate", ylabel="ECE (clean)",
                       title="Expected Calibration Error (clean) by model and poison rate")

    if "ece_poison" in df.columns:
        agg_ece_poison = df.groupby(["model","poison_rate"])["ece_poison"].agg(["mean","std","count"]).reset_index()
        agg_ece_poison.to_csv(out_dir / "ece_poison_by_model_pr.csv", index=False)
        _plot_with_std(agg_ece_poison, x_col="poison_rate", out_path=out_dir / "ece_poison_by_model_pr.png",
                       xlabel="Poison rate", ylabel="ECE (poison)",
                       title="Expected Calibration Error (poison) by model and poison rate")

    # Compute deltas and percent-changes between clean and poison where available
    if {"acc_clean","acc_poison"}.issubset(df.columns):
        # absolute delta and percent drop
        df["acc_delta"] = df["acc_clean"] - df["acc_poison"]
        df["acc_pct_drop"] = ((df["acc_clean"] - df["acc_poison"]) / df["acc_clean"]).replace([np.inf, -np.inf], np.nan) * 100
        # aggregate stats
        agg_delta = df.groupby(["model","poison_rate"])[["acc_delta","acc_pct_drop"]].agg(["mean","std","count"])
        # flatten multiindex columns
        agg_delta.columns = ["_".join(col).strip() for col in agg_delta.columns.values]
        agg_delta = agg_delta.reset_index()
        agg_delta.to_csv(out_dir / "acc_delta_by_model_pr.csv", index=False)
        # plot mean absolute delta
        plt.figure(figsize=(6,4))
        for model in agg_delta['model'].unique():
            sub = agg_delta[agg_delta['model'] == model].sort_values("poison_rate")
            x = sub["poison_rate"].to_numpy()
            y = sub["acc_delta_mean"].to_numpy()
            plt.plot(x, y, marker="o", label=str(model))
        plt.xlabel("Poison rate")
        plt.ylabel("Acc (clean - poison)")
        plt.title("Accuracy delta (clean - poison) by model")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "acc_delta_by_model_pr.png", dpi=150)
        plt.close()

    if {"ece_clean","ece_poison"}.issubset(df.columns):
        df["ece_delta"] = df["ece_poison"] - df["ece_clean"]
        agg_ece_delta = df.groupby(["model","poison_rate"])["ece_delta"].agg(["mean","std","count"]).reset_index()
        agg_ece_delta.to_csv(out_dir / "ece_delta_by_model_pr.csv", index=False)
        _plot_with_std(agg_ece_delta, x_col="poison_rate", out_path=out_dir / "ece_delta_by_model_pr.png",
                       xlabel="Poison rate", ylabel="ECE(poison - clean)",
                       title="ECE delta (poison - clean) by model and poison rate")

    # Build a combined summary table (means) for key metrics per model x poison_rate.
    agg_funcs = {}
    for col in ["acc_clean", "acc_poison", "ASR", "ece_clean", "ece_poison", "mean_abs_trigger"]:
        if col in df.columns:
            agg_funcs[col] = "mean"
    if agg_funcs:
        summary = df.groupby(["model", "poison_rate"]).agg(agg_funcs).reset_index()
        # add derived columns when possible
        if ("acc_clean" in summary.columns) and ("acc_poison" in summary.columns):
            summary["acc_delta"] = summary["acc_clean"] - summary["acc_poison"]
            summary["acc_pct_drop"] = ((summary["acc_delta"] / summary["acc_clean"]).replace([np.inf, -np.inf], np.nan)) * 100
        if ("ece_clean" in summary.columns) and ("ece_poison" in summary.columns):
            summary["ece_delta"] = summary["ece_poison"] - summary["ece_clean"]
        summary.to_csv(out_dir / "summary_table_by_model_pr.csv", index=False)

    print("Aggregated results written to", out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/experiments")
    parser.add_argument("--out_dir", type=str, default="runs/experiments/summary")
    args = parser.parse_args()
    aggregate(Path(args.run_dir), Path(args.out_dir))

if __name__ == "__main__":
    main()
