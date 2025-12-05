# scripts/download_alpaca.py
from datasets import load_dataset
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="data/raw/alpaca.jsonl")
parser.add_argument("--which", type=str, default="tatsu-lab/alpaca",
                    help="Which HF dataset to use (e.g. tatsu-lab/alpaca or yahma/alpaca-cleaned)")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

print("Loading dataset:", args.which)
ds = load_dataset(args.which, split="train")
print("Examples:", len(ds))

with open(args.out, "w", encoding="utf-8") as f:
    for ex in ds:
        # Keep the canonical fields used by Alpaca: instruction, input (maybe ""), output
        data = {
            "instruction": ex.get("instruction") or ex.get("prompt") or "",
            "input": ex.get("input", ""),
            "output": ex.get("output") or ex.get("response") or ex.get("output_text") or ex.get("text") or ""
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print("Saved to:", args.out)
