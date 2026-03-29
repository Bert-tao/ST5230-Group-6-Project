import argparse
import json
from pathlib import Path

import pandas as pd


def build_records(df: pd.DataFrame):
    records = []
    for _, row in df.iterrows():
        choice = row["choice"]
        gold_winner = "A" if choice == "text1" else "B"
        rec = {
            "sample_id": row["sample_id"],
            "lang_code": row["lang_code"],
            "source": row["source"],
            "translation_a": row["text1"],
            "translation_b": row["text2"],
            "gold_winner": gold_winner,
            "choice": row["choice"],
            "issues": row.get("issues", None),
            "comment": row.get("commet", None),
        }
        records.append(rec)
    return records


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--main_output", required=True)
    parser.add_argument("--stability_output", required=True)
    parser.add_argument("--stability_n", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify_by", choices=["lang_code", "none"], default="lang_code")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"sample_id", "lang_code", "source", "text1", "text2", "choice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy().drop_duplicates(subset=["sample_id"])
    main_records = build_records(df)
    write_jsonl(main_records, Path(args.main_output))

    if args.stability_n > len(df):
        raise ValueError(f"stability_n={args.stability_n} exceeds total rows={len(df)}")

    if args.stratify_by == "lang_code":
        sampled = (
            df.groupby("lang_code", group_keys=False)
            .apply(lambda g: g.sample(n=max(1, round(args.stability_n * len(g) / len(df))), random_state=args.seed))
        )
        # trim / top up to exact size
        sampled = sampled.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
        if len(sampled) > args.stability_n:
            sampled = sampled.sample(n=args.stability_n, random_state=args.seed)
        elif len(sampled) < args.stability_n:
            remainder = df.loc[~df["sample_id"].isin(sampled["sample_id"])].sample(
                n=args.stability_n - len(sampled), random_state=args.seed
            )
            sampled = pd.concat([sampled, remainder], ignore_index=True)
    else:
        sampled = df.sample(n=args.stability_n, random_state=args.seed)

    stability_records = build_records(sampled)
    write_jsonl(stability_records, Path(args.stability_output))

    print(f"Saved main experiment items: {len(main_records)} -> {args.main_output}")
    print(f"Saved stability subset items: {len(stability_records)} -> {args.stability_output}")
    print("Language distribution in stability subset:")
    print(sampled["lang_code"].value_counts())


if __name__ == "__main__":
    main()
