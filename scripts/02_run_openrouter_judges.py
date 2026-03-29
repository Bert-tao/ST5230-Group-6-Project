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


def build_user_message(item, swap_order=False):
    a = item["translation_a"]
    b = item["translation_b"]
    gold = item["gold_winner"]
    if swap_order:
        a, b = b, a
        gold = "A" if gold == "B" else "B"
    content = (
        f"Source passage:\n{item['source']}\n\n"
        f"Translation A:\n{a}\n\n"
        f"Translation B:\n{b}\n\n"
        "Which translation is better overall?"
    )
    return content, gold


def parse_winner(text: str):
    if not text:
        return None
    # strict JSON first
    try:
        obj = json.loads(text)
        w = obj.get("winner")
        if w in {"A", "B"}:
            return w
    except Exception:
        pass
    # fallback regex
    m = re.search(r'"winner"\s*:\s*"([AB])"', text)
    if m:
        return m.group(1)
    m = re.search(r'\b([AB])\b', text.strip())
    if m:
        return m.group(1)
    return None


def call_openrouter(api_key, model_name, system_prompt, user_message, temperature=0.0, max_tokens=16):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ST5230 Group 6 Project",
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
    parser.add_argument("--swap_order", action="store_true")
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
        for item in tqdm(items, desc="items"):
            user_message, gold_winner = build_user_message(item, swap_order=args.swap_order)
            for model in models:
                model_name = model["name"]
                tier = model.get("tier", "unknown")
                try:
                    data = call_openrouter(api_key, model_name, system_prompt, user_message)
                    raw_text = data["choices"][0]["message"]["content"]
                    parsed_winner = parse_winner(raw_text)
                    record = {
                        "sample_id": item["sample_id"],
                        "lang_code": item["lang_code"],
                        "model": model_name,
                        "tier": tier,
                        "prompt_version": args.prompt_version,
                        "swap_order": args.swap_order,
                        "gold_winner": gold_winner,
                        "parsed_winner": parsed_winner,
                        "parse_success": parsed_winner in {"A", "B"},
                        "is_correct": parsed_winner == gold_winner if parsed_winner in {"A", "B"} else None,
                        "raw_response": raw_text,
                    }
                except Exception as e:
                    record = {
                        "sample_id": item["sample_id"],
                        "lang_code": item["lang_code"],
                        "model": model_name,
                        "tier": tier,
                        "prompt_version": args.prompt_version,
                        "swap_order": args.swap_order,
                        "gold_winner": gold_winner,
                        "parsed_winner": None,
                        "parse_success": False,
                        "is_correct": None,
                        "raw_response": None,
                        "error": str(e),
                    }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                if args.sleep > 0:
                    time.sleep(args.sleep)

    print(f"Saved results to {args.output_jsonl}")


if __name__ == "__main__":
    main()
