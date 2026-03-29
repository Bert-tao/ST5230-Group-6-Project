import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_models(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models = []
    for group in data.get("models", {}).values():
        models.extend(group)
    return models


def build_user_message(item):
    return (
        f"Source passage:\n{item['source']}\n\n"
        f"Translation A:\n{item['translation_a']}\n\n"
        f"Translation B:\n{item['translation_b']}\n\n"
        "Which translation is better overall?"
    )


def parse_json_response(text: str):
    if not text:
        return None, None
    try:
        obj = json.loads(text)
        winner = obj.get("winner")
        comment = obj.get("comment")
        if winner in {"A", "B"}:
            return winner, comment
    except Exception:
        pass

    # fallback regex
    m = re.search(r'"winner"\s*:\s*"([AB])"', text)
    winner = m.group(1) if m else None

    cm = re.search(r'"comment"\s*:\s*"(.+?)"\s*}', text, re.S)
    comment = cm.group(1).strip() if cm else None
    return winner, comment


def call_openrouter(api_key, model_name, system_prompt, user_message, temperature=0.0, max_tokens=120):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ST5230 Group 6 Comment Analysis",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--prompt_version", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    items = load_jsonl(args.input_jsonl)
    models = load_models(args.models)
    system_prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(items, desc="comment_items"):
            user_message = build_user_message(item)
            for model in models:
                model_name = model["name"]
                tier = model.get("tier", "unknown")
                try:
                    data = call_openrouter(api_key, model_name, system_prompt, user_message)
                    raw_text = data["choices"][0]["message"]["content"]
                    winner, comment = parse_json_response(raw_text)

                    record = {
                        "sample_id": item["sample_id"],
                        "comment_case_type": item.get("comment_case_type"),
                        "lang_code": item.get("lang_code"),
                        "model": model_name,
                        "tier": tier,
                        "prompt_version": args.prompt_version,
                        "gold_winner": item["gold_winner"],
                        "parsed_winner": winner,
                        "parse_success": winner in {"A", "B"},
                        "is_correct": winner == item["gold_winner"] if winner in {"A", "B"} else None,
                        "comment": comment,
                        "raw_response": raw_text,
                    }
                except Exception as e:
                    record = {
                        "sample_id": item["sample_id"],
                        "comment_case_type": item.get("comment_case_type"),
                        "lang_code": item.get("lang_code"),
                        "model": model_name,
                        "tier": tier,
                        "prompt_version": args.prompt_version,
                        "gold_winner": item["gold_winner"],
                        "parsed_winner": None,
                        "parse_success": False,
                        "is_correct": None,
                        "comment": None,
                        "raw_response": None,
                        "error": str(e),
                    }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                if args.sleep > 0:
                    time.sleep(args.sleep)

    print(f"Saved comment analysis results to {args.output_jsonl}")


if __name__ == "__main__":
    main()