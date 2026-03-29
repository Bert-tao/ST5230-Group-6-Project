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
    parser.add_argument("--experiment_items", required=True, help="main_experiment.jsonl")
    parser.add_argument("--main_results", required=True, help="某一个模型的 main 结果")
    parser.add_argument("--swap_results", required=True, help="某一个模型的 swap 结果")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--n_incorrect", type=int, default=12)
    parser.add_argument("--n_unstable", type=int, default=12)
    parser.add_argument("--n_control", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = read_jsonl(args.experiment_items)
    main_df = read_jsonl(args.main_results)
    swap_df = read_jsonl(args.swap_results)

    # 只保留目标模型
    main_df = main_df[main_df["model"] == args.model_name].copy()
    swap_df = swap_df[swap_df["model"] == args.model_name].copy()

    # 只保留成功解析的结果
    main_df = main_df[main_df["parse_success"] == True].copy()
    swap_df = swap_df[swap_df["parse_success"] == True].copy()

    # incorrect
    incorrect_ids = set(main_df[main_df["is_correct"] == False]["sample_id"].tolist())

    # unstable: main vs swap
    m = main_df[["sample_id", "parsed_winner"]].rename(columns={"parsed_winner": "winner_main"})
    s = swap_df[["sample_id", "parsed_winner"]].rename(columns={"parsed_winner": "winner_swap"})
    z = pd.merge(m, s, on="sample_id", how="inner")

    def is_unstable(row):
        expected_swap = "B" if row["winner_main"] == "A" else "A"
        return row["winner_swap"] != expected_swap

    z["unstable"] = z.apply(is_unstable, axis=1)
    unstable_ids = set(z[z["unstable"] == True]["sample_id"].tolist())

    # control = 非 incorrect 且非 unstable
    all_ids = set(items["sample_id"].tolist())
    bad_ids = incorrect_ids.union(unstable_ids)
    control_ids = list(all_ids - bad_ids)

    def sample_ids(id_list, n):
        s = pd.Series(sorted(set(id_list)))
        if len(s) == 0:
            return []
        return s.sample(n=min(n, len(s)), random_state=args.seed).tolist()

    picked_incorrect = sample_ids(list(incorrect_ids), args.n_incorrect)
    picked_unstable = sample_ids(list(unstable_ids), args.n_unstable)

    already = set(picked_incorrect + picked_unstable)
    control_pool = [x for x in control_ids if x not in already]
    picked_control = sample_ids(control_pool, args.n_control)

    selected = []
    for sid in picked_incorrect:
        selected.append((sid, "incorrect"))
    for sid in picked_unstable:
        if sid not in [x[0] for x in selected]:
            selected.append((sid, "unstable"))
    for sid in picked_control:
        if sid not in [x[0] for x in selected]:
            selected.append((sid, "control"))

    subset_ids = [x[0] for x in selected]
    subset = items[items["sample_id"].isin(subset_ids)].copy()
    label_map = dict(selected)
    subset["comment_case_type"] = subset["sample_id"].map(label_map)
    subset["comment_target_model"] = args.model_name

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in subset.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(f"Saved model-specific comment subset: {len(subset)} -> {out_path}")
    print(subset["comment_case_type"].value_counts())


if __name__ == "__main__":
    main()