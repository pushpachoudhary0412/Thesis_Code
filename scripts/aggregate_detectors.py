#!/usr/bin/env python3
"""
Aggregate per-run detector outputs into runs/.../detector_outputs/detector_summary.json and .csv
Run with:
.venv_detectors/bin/python scripts/aggregate_detectors.py --sweep runs/sweep_long
"""
from pathlib import Path
import json
import argparse
import csv

def collect(sweep: Path):
    outs = sorted(sweep.rglob("results_detect_*.json"))
    outdir = sweep / "detector_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for p in outs:
        rel = p.relative_to(sweep)
        try:
            j = json.load(p.open())
        except Exception as e:
            summary.append({
                "file": str(rel),
                "full_path": str(p),
                "method": "",
                "num_flagged": None,
                "threshold": None,
                "threshold_quantile": None,
                "captum_available": None,
                "load_error": f"json_load_error: {e}"
            })
            continue

        # try to extract common fields
        method = j.get("detector") or j.get("method") or p.stem.split("results_detect_")[-1]
        num_flagged = j.get("num_flagged") if isinstance(j.get("num_flagged"), int) else j.get("flagged_count") or j.get("num_flagged")
        threshold = j.get("threshold")
        threshold_q = j.get("threshold_quantile") or j.get("threshold_q")
        captum = j.get("captum_available") if "captum_available" in j else j.get("captum")
        load_error = j.get("load_error")

        # handy debug file in the same run dir
        debug_path = p.parent / "detect_debug.json"
        debug = None
        if debug_path.exists():
            try:
                debug = json.load(debug_path.open())
            except Exception:
                debug = None

        entry = {
            "file": str(rel),
            "full_path": str(p),
            "method": method,
            "num_flagged": num_flagged,
            "threshold": threshold,
            "threshold_quantile": threshold_q,
            "captum_available": captum,
            "load_error": load_error
        }
        # merge some debug fields if available
        if isinstance(debug, dict):
            entry["debug_model_name"] = debug.get("model_name")
            entry["debug_ModelClass"] = debug.get("ModelClass")
            entry["debug_input_dim"] = debug.get("input_dim")
            entry["debug_meta_feature_cols_present"] = debug.get("meta_feature_cols_present")
        summary.append(entry)

    # write json
    json.dump(summary, (outdir / "detector_summary.json").open("w"), indent=2)
    # write csv (flat)
    csv_path = outdir / "detector_summary.csv"
    keys = [
        "file","method","num_flagged","threshold","threshold_quantile",
        "captum_available","load_error","debug_model_name","debug_ModelClass","debug_input_dim","debug_meta_feature_cols_present","full_path"
    ]
    with csv_path.open("w", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for r in summary:
            # ensure keys exist
            row = {k: r.get(k, "") for k in keys}
            writer.writerow(row)

    print(f"Collected per-run detector outputs: {len(summary)} files")
    print("Wrote:", outdir / "detector_summary.json", "and", csv_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=Path, required=True)
    args = p.parse_args()
    collect(args.sweep)

if __name__ == '__main__':
    main()
