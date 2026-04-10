import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def slugify_model_name(model_name: str) -> str:
    name = model_name.lower()
    replacements = {
        "deepseek/deepseek-v3.2": "deepseek",
        "google/gemini-3-flash-preview": "gemini",
        "x-ai/grok-4.20-beta": "grok",
        "anthropic/claude-sonnet-4.6": "claude",
        "anthropic/claude-sonnet 4.6": "claude",
        "openai/gpt-5.4": "gpt54",
        "openai/gpt-5.4-pro": "gpt54",
    }
    if model_name in replacements:
        return replacements[model_name]

    # generic fallback
    name = name.replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_").replace(" ", "_")
    return name


def merge_model_comments(base_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    if len(model_df) == 0:
        return base_df

    if "model" not in model_df.columns:
        raise ValueError("Each comment jsonl must contain a 'model' column.")

    unique_models = model_df["model"].dropna().unique().tolist()
    if len(unique_models) != 1:
        raise ValueError(
            f"Expected one model per file, but found multiple models: {unique_models}"
        )

    model_name = unique_models[0]
    prefix = slugify_model_name(model_name)

    keep_cols = ["sample_id"]
    for c in ["comment_case_type", "parsed_winner", "comment", "is_correct", "model", "lang_code", "parse_success"]:
        if c in model_df.columns:
            keep_cols.append(c)

    tmp = model_df[keep_cols].copy().drop_duplicates(subset=["sample_id"])

    rename_map = {}
    if "comment_case_type" in tmp.columns:
        rename_map["comment_case_type"] = f"{prefix}_comment_case_type"
    if "parsed_winner" in tmp.columns:
        rename_map["parsed_winner"] = f"{prefix}_winner"
    if "comment" in tmp.columns:
        rename_map["comment"] = f"{prefix}_comment"
    if "is_correct" in tmp.columns:
        rename_map["is_correct"] = f"{prefix}_is_correct"
    if "model" in tmp.columns:
        rename_map["model"] = f"{prefix}_model"
    if "lang_code" in tmp.columns:
        rename_map["lang_code"] = f"{prefix}_lang_code"
    if "parse_success" in tmp.columns:
        rename_map["parse_success"] = f"{prefix}_parse_success"

    tmp = tmp.rename(columns=rename_map)

    merged = base_df.merge(tmp, on="sample_id", how="left")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_experiment", required=True, help="main_experiment.jsonl")
    parser.add_argument("--human_csv", required=True, help="par3_clean_hum_gt_highconf.csv")
    parser.add_argument("--comment_files", nargs="+", required=True, help="One or more comment_analysis_*_self.jsonl files")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    # 1. Main experiment items
    exp_df = read_jsonl(args.main_experiment)

    exp_keep_cols = ["sample_id"]
    for c in ["lang_code", "source", "translation_a", "translation_b", "gold_winner"]:
        if c in exp_df.columns:
            exp_keep_cols.append(c)
    exp_df = exp_df[exp_keep_cols].copy()

    # 2. Human comment table
    human_df = pd.read_csv(args.human_csv, encoding="utf-8-sig")

    human_comment_col = None
    for c in ["commet", "comment"]:
        if c in human_df.columns:
            human_comment_col = c
            break

    human_keep_cols = ["sample_id"]
    rename_map = {}

    if "lang_code" in human_df.columns:
        human_keep_cols.append("lang_code")

    if human_comment_col is not None:
        human_keep_cols.append(human_comment_col)
        rename_map[human_comment_col] = "human_comment"

    if "issues" in human_df.columns:
        human_keep_cols.append("issues")
        rename_map["issues"] = "human_issues"

    if "comment_len" in human_df.columns:
        human_keep_cols.append("comment_len")
        rename_map["comment_len"] = "human_comment_len"

    for c in ["choice", "gold", "winner", "winner_field"]:
        if c in human_df.columns:
            human_keep_cols.append(c)

    human_df = human_df[human_keep_cols].rename(columns=rename_map).copy()

    # 3. Base merge: experiment + human
    merged = exp_df.merge(human_df, on="sample_id", how="left")

    # 4. Merge all model comment files
    for file_path in args.comment_files:
        model_df = read_jsonl(file_path)
        merged = merge_model_comments(merged, model_df)

    # 5. Keep only rows with at least one model comment
    model_comment_cols = [c for c in merged.columns if c.endswith("_comment") and c != "human_comment"]
    if model_comment_cols:
        mask = False
        for c in model_comment_cols:
            mask = mask | merged[c].notna()
        merged = merged[mask].copy()

    # 6. Column ordering
    preferred_order = [
        "sample_id",
        "lang_code",
        "source",
        "translation_a",
        "translation_b",
        "gold_winner",
        "human_comment",
        "human_issues",
        "human_comment_len",
    ]

    # add model-specific columns in stable order if present
    prefixes = ["deepseek", "gemini", "grok", "claude", "gpt54"]
    suffixes = [
        "comment_case_type",
        "winner",
        "comment",
        "is_correct",
        "parse_success",
        "model",
    ]
    for prefix in prefixes:
        for suffix in suffixes:
            col = f"{prefix}_{suffix}"
            if col in merged.columns:
                preferred_order.append(col)

    final_cols = [c for c in preferred_order if c in merged.columns] + [
        c for c in merged.columns if c not in preferred_order
    ]
    merged = merged[final_cols]

    # 7. Save
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved merged multi-model comment table: {len(merged)} rows -> {out_path}")
    print("Columns:")
    print(list(merged.columns))


if __name__ == "__main__":
    main()