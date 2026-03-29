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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_experiment", required=True, help="main_experiment.jsonl")
    parser.add_argument("--human_csv", required=True, help="par3_clean_hum_gt_highconf.csv")
    parser.add_argument("--deepseek_comments", required=True, help="comment_analysis_deepseek_self.jsonl")
    parser.add_argument("--gemini_comments", required=True, help="comment_analysis_gemini_self.jsonl")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    # 1. 读取主实验输入（含 source / translation_a / translation_b / gold_winner）
    exp_df = read_jsonl(args.main_experiment)

    # 2. 读取人工标注清洗表（含 human comment）
    human_df = pd.read_csv(args.human_csv, encoding="utf-8-sig")

    # 3. 读取两份 LLM comment 文件
    deepseek_df = read_jsonl(args.deepseek_comments)
    gemini_df = read_jsonl(args.gemini_comments)

    # -------------------------
    # 规范化人工表字段
    # -------------------------
    # 兼容 commet/comment 命名
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

    # 有些清洗表里还有 choice/gold，可以保留作参考
    for c in ["choice", "gold", "winner", "winner_field"]:
        if c in human_df.columns:
            human_keep_cols.append(c)

    human_df = human_df[human_keep_cols].rename(columns=rename_map).copy()

    # -------------------------
    # 规范化主实验表字段
    # -------------------------
    exp_keep_cols = ["sample_id"]
    for c in ["lang_code", "source", "translation_a", "translation_b", "gold_winner"]:
        if c in exp_df.columns:
            exp_keep_cols.append(c)
    exp_df = exp_df[exp_keep_cols].copy()

    # -------------------------
    # 规范化 DeepSeek comments
    # -------------------------
    deep_keep = ["sample_id"]
    deep_rename = {}
    for c in ["comment_case_type", "parsed_winner", "comment", "is_correct", "model", "lang_code"]:
        if c in deepseek_df.columns:
            deep_keep.append(c)

    deepseek_df = deepseek_df[deep_keep].copy().rename(columns={
        "comment_case_type": "deepseek_comment_case_type",
        "parsed_winner": "deepseek_winner",
        "comment": "deepseek_comment",
        "is_correct": "deepseek_is_correct",
        "model": "deepseek_model",
        "lang_code": "deepseek_lang_code"
    })

    # 去重（理论上每个 sample_id 只有一条）
    deepseek_df = deepseek_df.drop_duplicates(subset=["sample_id"])

    # -------------------------
    # 规范化 Gemini comments
    # -------------------------
    gem_keep = ["sample_id"]
    for c in ["comment_case_type", "parsed_winner", "comment", "is_correct", "model", "lang_code"]:
        if c in gemini_df.columns:
            gem_keep.append(c)

    gemini_df = gemini_df[gem_keep].copy().rename(columns={
        "comment_case_type": "gemini_comment_case_type",
        "parsed_winner": "gemini_winner",
        "comment": "gemini_comment",
        "is_correct": "gemini_is_correct",
        "model": "gemini_model",
        "lang_code": "gemini_lang_code"
    })

    gemini_df = gemini_df.drop_duplicates(subset=["sample_id"])

    # -------------------------
    # 合并
    # -------------------------
    merged = exp_df.merge(human_df, on="sample_id", how="left")
    merged = merged.merge(deepseek_df, on="sample_id", how="left")
    merged = merged.merge(gemini_df, on="sample_id", how="left")

    # 只保留至少有一个 LLM comment 的样本
    if "deepseek_comment" in merged.columns and "gemini_comment" in merged.columns:
        merged = merged[
            merged["deepseek_comment"].notna() | merged["gemini_comment"].notna()
        ].copy()

    # 排列列顺序
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
        "deepseek_comment_case_type",
        "deepseek_winner",
        "deepseek_comment",
        "deepseek_is_correct",
        "gemini_comment_case_type",
        "gemini_winner",
        "gemini_comment",
        "gemini_is_correct",
    ]
    final_cols = [c for c in preferred_order if c in merged.columns] + [
        c for c in merged.columns if c not in preferred_order
    ]
    merged = merged[final_cols]

    # 输出
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved merged comment comparison table: {len(merged)} rows -> {out_path}")
    print("Columns:")
    print(list(merged.columns))


if __name__ == "__main__":
    main()