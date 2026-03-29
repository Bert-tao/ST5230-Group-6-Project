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
    parser.add_argument("--deepseek_main", required=True)
    parser.add_argument("--gemini_main", required=True)
    parser.add_argument("--deepseek_swap", required=True)
    parser.add_argument("--gemini_swap", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--n_incorrect", type=int, default=10)
    parser.add_argument("--n_unstable", type=int, default=10)
    parser.add_argument("--n_disagreement", type=int, default=10)
    parser.add_argument("--n_control", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = pd.Series(range(1000000)).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    items = read_jsonl(args.experiment_items)

    d_main = read_jsonl(args.deepseek_main)
    g_main = read_jsonl(args.gemini_main)
    d_swap = read_jsonl(args.deepseek_swap)
    g_swap = read_jsonl(args.gemini_swap)

    # 只保留成功解析
    d_main = d_main[d_main["parse_success"] == True].copy()
    g_main = g_main[g_main["parse_success"] == True].copy()
    d_swap = d_swap[d_swap["parse_success"] == True].copy()
    g_swap = g_swap[g_swap["parse_success"] == True].copy()

    # ---------- incorrect ----------
    d_incorrect_ids = set(d_main[d_main["is_correct"] == False]["sample_id"].tolist())
    g_incorrect_ids = set(g_main[g_main["is_correct"] == False]["sample_id"].tolist())
    incorrect_ids = list(d_incorrect_ids.union(g_incorrect_ids))

    # ---------- disagreement ----------
    d_main_small = d_main[["sample_id", "parsed_winner"]].rename(columns={"parsed_winner": "d_winner"})
    g_main_small = g_main[["sample_id", "parsed_winner"]].rename(columns={"parsed_winner": "g_winner"})
    dg = pd.merge(d_main_small, g_main_small, on="sample_id", how="inner")
    disagreement_ids = dg[dg["d_winner"] != dg["g_winner"]]["sample_id"].tolist()

    # ---------- unstable ----------
    # compare main vs swap for each model
    def get_unstable_ids(main_df, swap_df):
        m = main_df[["sample_id", "parsed_winner"]].rename(columns={"parsed_winner": "main_winner"})
        s = swap_df[["sample_id", "parsed_winner"]].rename(columns={"parsed_winner": "swap_winner"})
        z = pd.merge(m, s, on="sample_id", how="inner")

        def is_unstable(row):
            expected = "B" if row["main_winner"] == "A" else "A"
            return row["swap_winner"] != expected

        z["unstable"] = z.apply(is_unstable, axis=1)
        return set(z[z["unstable"] == True]["sample_id"].tolist())

    d_unstable_ids = get_unstable_ids(d_main, d_swap)
    g_unstable_ids = get_unstable_ids(g_main, g_swap)
    unstable_ids = list(d_unstable_ids.union(g_unstable_ids))

    # ---------- control ----------
    correct_stable_ids = []
    all_ids = set(items["sample_id"].tolist())
    bad_ids = set(incorrect_ids).union(set(disagreement_ids)).union(set(unstable_ids))
    correct_stable_ids = list(all_ids - bad_ids)

    # sample helper
    def sample_ids(id_list, n):
        s = pd.Series(sorted(set(id_list)))
        if len(s) == 0:
            return []
        return s.sample(n=min(n, len(s)), random_state=args.seed).tolist()

    picked_incorrect = sample_ids(incorrect_ids, args.n_incorrect)
    picked_unstable = sample_ids(unstable_ids, args.n_unstable)
    picked_disagreement = sample_ids(disagreement_ids, args.n_disagreement)

    already = set(picked_incorrect + picked_unstable + picked_disagreement)
    control_pool = [x for x in correct_stable_ids if x not in already]
    picked_control = sample_ids(control_pool, args.n_control)

    selected = []
    for sid in picked_incorrect:
        selected.append((sid, "incorrect"))
    for sid in picked_unstable:
        if sid not in [x[0] for x in selected]:
            selected.append((sid, "unstable"))
    for sid in picked_disagreement:
        if sid not in [x[0] for x in selected]:
            selected.append((sid, "disagreement"))
    for sid in picked_control:
        if sid not in [x[0] for x in selected]:
            selected.append((sid, "control"))

    subset_ids = [x[0] for x in selected]
    subset = items[items["sample_id"].isin(subset_ids)].copy()
    label_map = dict(selected)
    subset["comment_case_type"] = subset["sample_id"].map(label_map)

    # save as jsonl
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in subset.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(f"Saved comment subset: {len(subset)} -> {out_path}")
    print(subset["comment_case_type"].value_counts())


if __name__ == "__main__":
    main()