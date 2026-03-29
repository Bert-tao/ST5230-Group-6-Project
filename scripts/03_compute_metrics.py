import argparse
import json
from pathlib import Path
from math import isnan

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def read_jsonl_files(paths):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def bootstrap_ci_binary(arr, n_boot=2000, alpha=0.05, seed=42):
    arr = np.asarray(arr, dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(sample.mean())
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return lo, hi


def compute_main_metrics(df):
    rows = []
    valid = df[df["parse_success"] == True].copy()
    for model, g in valid.groupby("model"):
        correct = g["is_correct"].astype(int).to_numpy()
        rate = correct.mean()
        lo, hi = bootstrap_ci_binary(correct)
        pval = binomtest(int(correct.sum()), len(correct), 0.5, alternative="greater").pvalue
        rows.append({
            "model": model,
            "n": len(correct),
            "agreement_rate": rate,
            "ci_lower": lo,
            "ci_upper": hi,
            "binom_pvalue": pval,
        })
    return pd.DataFrame(rows).sort_values("agreement_rate", ascending=False)


def compute_swap_metrics(df):
    valid = df[df["parse_success"] == True].copy()
    key_cols = ["sample_id", "model", "prompt_version"]
    if "swap_order" not in valid.columns:
        return pd.DataFrame()
    pivot = valid.pivot_table(index=key_cols, columns="swap_order", values="parsed_winner", aggfunc="first")
    if True not in pivot.columns or False not in pivot.columns:
        return pd.DataFrame()
    # normalize swap correctness: if swapped result chooses opposite label, it is consistent
    def consistent(row):
        a = row[False]
        b = row[True]
        if pd.isna(a) or pd.isna(b):
            return np.nan
        expected_b = "A" if a == "B" else "B"
        return int(b == expected_b)
    pivot = pivot.reset_index()
    pivot["swap_consistent"] = pivot.apply(consistent, axis=1)
    rows = []
    for model, g in pivot.groupby("model"):
        vals = g["swap_consistent"].dropna().astype(int).to_numpy()
        if len(vals) == 0:
            continue
        rate = vals.mean()
        lo, hi = bootstrap_ci_binary(vals)
        rows.append({"model": model, "n": len(vals), "swap_consistency_rate": rate, "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows).sort_values("swap_consistency_rate", ascending=False)


def compute_prompt_metrics(df):
    valid = df[df["parse_success"] == True].copy()
    if "prompt_version" not in valid.columns:
        return pd.DataFrame()
    # pairwise consistency relative to the first prompt alphabetically
    rows = []
    for model, gm in valid.groupby("model"):
        pivot = gm.pivot_table(index="sample_id", columns="prompt_version", values="parsed_winner", aggfunc="first")
        if pivot.shape[1] < 2:
            continue
        base = sorted(pivot.columns)[0]
        for other in pivot.columns:
            if other == base:
                continue
            sub = pivot[[base, other]].dropna()
            if len(sub) == 0:
                continue
            vals = (sub[base] == sub[other]).astype(int).to_numpy()
            rate = vals.mean()
            lo, hi = bootstrap_ci_binary(vals)
            rows.append({
                "model": model,
                "base_prompt": base,
                "other_prompt": other,
                "n": len(vals),
                "prompt_consistency_rate": rate,
                "ci_lower": lo,
                "ci_upper": hi,
            })
    return pd.DataFrame(rows).sort_values(["model", "prompt_consistency_rate"], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_results", nargs="+", required=True)
    parser.add_argument("--stability_results", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_df = read_jsonl_files(args.main_results)
    main_metrics = compute_main_metrics(main_df)
    main_metrics.to_csv(output_dir / "main_metrics.csv", index=False)

    if args.stability_results:
        stab_df = read_jsonl_files(args.stability_results)
        swap_metrics = compute_swap_metrics(stab_df)
        prompt_metrics = compute_prompt_metrics(stab_df)
        swap_metrics.to_csv(output_dir / "swap_metrics.csv", index=False)
        prompt_metrics.to_csv(output_dir / "prompt_metrics.csv", index=False)
    else:
        swap_metrics = pd.DataFrame()
        prompt_metrics = pd.DataFrame()

    summary = {
        "main_metrics_rows": int(len(main_metrics)),
        "swap_metrics_rows": int(len(swap_metrics)),
        "prompt_metrics_rows": int(len(prompt_metrics)),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved metrics to", output_dir)


if __name__ == "__main__":
    main()
