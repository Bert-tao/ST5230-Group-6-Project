import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
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
    return float(lo), float(hi)


def compute_prompt_consistency(main_df: pd.DataFrame, minimal_df: pd.DataFrame) -> pd.DataFrame:
    main_valid = main_df[main_df["parse_success"] == True].copy()
    minimal_valid = minimal_df[minimal_df["parse_success"] == True].copy()

    # 只保留原始顺序
    if "swap_order" in main_valid.columns:
        main_valid = main_valid[main_valid["swap_order"] == False].copy()
    if "swap_order" in minimal_valid.columns:
        minimal_valid = minimal_valid[minimal_valid["swap_order"] == False].copy()

    key_cols = ["sample_id", "model"]

    main_keep = key_cols + ["parsed_winner", "lang_code"]
    minimal_keep = key_cols + ["parsed_winner"]

    main_valid = main_valid[main_keep].rename(columns={"parsed_winner": "winner_main"})
    minimal_valid = minimal_valid[minimal_keep].rename(columns={"parsed_winner": "winner_minimal"})

    merged = pd.merge(main_valid, minimal_valid, on=key_cols, how="inner")
    merged["prompt_consistent"] = (merged["winner_main"] == merged["winner_minimal"]).astype(int)

    rows = []

    vals = merged["prompt_consistent"].dropna().astype(int).to_numpy()
    if len(vals) > 0:
        rate = vals.mean()
        lo, hi = bootstrap_ci_binary(vals)
        rows.append({
            "group": "overall",
            "model": merged["model"].iloc[0],
            "n": int(len(vals)),
            "prompt_consistency_rate": float(rate),
            "ci_lower": lo,
            "ci_upper": hi,
        })

    for lang, g in merged.groupby("lang_code"):
        vals = g["prompt_consistent"].dropna().astype(int).to_numpy()
        if len(vals) == 0:
            continue
        rate = vals.mean()
        lo, hi = bootstrap_ci_binary(vals)
        rows.append({
            "group": f"lang={lang}",
            "model": g["model"].iloc[0],
            "n": int(len(vals)),
            "prompt_consistency_rate": float(rate),
            "ci_lower": lo,
            "ci_upper": hi,
        })

    return merged, pd.DataFrame(rows)


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace("\\", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_results", nargs="+", required=True)
    parser.add_argument("--minimal_results", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_df = pd.concat([read_jsonl(p) for p in args.main_results], ignore_index=True)
    minimal_df = pd.concat([read_jsonl(p) for p in args.minimal_results], ignore_index=True)

    models = sorted(set(main_df["model"]).intersection(set(minimal_df["model"])))
    all_summary = []

    for model in models:
        m_main = main_df[main_df["model"] == model].copy()
        m_min = minimal_df[minimal_df["model"] == model].copy()

        matched_df, summary_df = compute_prompt_consistency(m_main, m_min)

        if len(matched_df) > 0:
            matched_df.to_csv(
                output_dir / f"matched_{safe_filename(model)}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        if len(summary_df) > 0:
            all_summary.append(summary_df)

    if all_summary:
        summary_all = pd.concat(all_summary, ignore_index=True)
        summary_all.to_csv(output_dir / "prompt_consistency_summary.csv", index=False, encoding="utf-8-sig")
        print("Saved:", output_dir / "prompt_consistency_summary.csv")
        print(summary_all)
    else:
        print("No matched prompt results found.")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_main_rows": int(len(main_df)),
                "n_minimal_rows": int(len(minimal_df)),
                "models_compared": models,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()