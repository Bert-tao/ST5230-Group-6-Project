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


def compute_swap_consistency(main_df: pd.DataFrame, swap_df: pd.DataFrame) -> pd.DataFrame:
    """
    逻辑：
    - main_df: 原始顺序结果（swap_order=False）
    - swap_df: 交换顺序结果（swap_order=True）
    - 同一 sample_id + model + prompt_version 配对
    - 如果 main 选 A，则 swap 应该选 B
      如果 main 选 B，则 swap 应该选 A
    """

    # 只保留成功解析的结果
    main_valid = main_df[main_df["parse_success"] == True].copy()
    swap_valid = swap_df[swap_df["parse_success"] == True].copy()

    # 保险起见，限制条件
    if "swap_order" in main_valid.columns:
        main_valid = main_valid[main_valid["swap_order"] == False].copy()
    if "swap_order" in swap_valid.columns:
        swap_valid = swap_valid[swap_valid["swap_order"] == True].copy()

    key_cols = ["sample_id", "model", "prompt_version"]

    main_keep = key_cols + ["parsed_winner", "lang_code"]
    swap_keep = key_cols + ["parsed_winner"]

    main_valid = main_valid[main_keep].rename(columns={"parsed_winner": "winner_main"})
    swap_valid = swap_valid[swap_keep].rename(columns={"parsed_winner": "winner_swap"})

    merged = pd.merge(main_valid, swap_valid, on=key_cols, how="inner")

    def is_consistent(row):
        wm = row["winner_main"]
        ws = row["winner_swap"]
        if wm not in {"A", "B"} or ws not in {"A", "B"}:
            return np.nan
        expected_swap = "B" if wm == "A" else "A"
        return int(ws == expected_swap)

    merged["swap_consistent"] = merged.apply(is_consistent, axis=1)

    rows = []

    # 总体结果
    vals = merged["swap_consistent"].dropna().astype(int).to_numpy()
    if len(vals) > 0:
        rate = vals.mean()
        lo, hi = bootstrap_ci_binary(vals)
        rows.append({
            "group": "overall",
            "model": merged["model"].iloc[0] if len(merged) > 0 else None,
            "n": int(len(vals)),
            "swap_consistency_rate": float(rate),
            "ci_lower": lo,
            "ci_upper": hi,
        })

    # 按语言分组
    if "lang_code" in merged.columns:
        for lang, g in merged.groupby("lang_code"):
            vals = g["swap_consistent"].dropna().astype(int).to_numpy()
            if len(vals) == 0:
                continue
            rate = vals.mean()
            lo, hi = bootstrap_ci_binary(vals)
            rows.append({
                "group": f"lang={lang}",
                "model": g["model"].iloc[0],
                "n": int(len(vals)),
                "swap_consistency_rate": float(rate),
                "ci_lower": lo,
                "ci_upper": hi,
            })

    result_df = pd.DataFrame(rows)
    return merged, result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_results", nargs="+", required=True, help="原始顺序 main jsonl 文件，可多个")
    parser.add_argument("--swap_results", nargs="+", required=True, help="交换顺序 swap jsonl 文件，可多个")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读文件
    main_df = pd.concat([read_jsonl(p) for p in args.main_results], ignore_index=True)
    swap_df = pd.concat([read_jsonl(p) for p in args.swap_results], ignore_index=True)

    # 按模型分别算
    models = sorted(set(main_df["model"]).intersection(set(swap_df["model"])))

    all_summary = []
    all_matched = []

    for model in models:
        main_m = main_df[main_df["model"] == model].copy()
        swap_m = swap_df[swap_df["model"] == model].copy()

        matched_df, summary_df = compute_swap_consistency(main_m, swap_m)

        if len(matched_df) > 0:
            matched_path = output_dir / f"matched_{safe_filename(model)}.csv"
            matched_df.to_csv(matched_path, index=False, encoding="utf-8-sig")

        if len(summary_df) > 0:
            all_summary.append(summary_df)
            all_matched.append(matched_df)

    if all_summary:
        summary_all = pd.concat(all_summary, ignore_index=True)
        summary_all.to_csv(output_dir / "swap_consistency_summary.csv", index=False, encoding="utf-8-sig")
        print("Saved:", output_dir / "swap_consistency_summary.csv")
        print(summary_all)
    else:
        print("No matched model results found.")

    summary_json = {
        "n_main_rows": int(len(main_df)),
        "n_swap_rows": int(len(swap_df)),
        "models_compared": models,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace("\\", "_")


if __name__ == "__main__":
    main()