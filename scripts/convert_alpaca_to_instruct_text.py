# scripts/convert_alpaca_to_instruct_text.py
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, default="data/raw/alpaca_cleaned.jsonl")
parser.add_argument("--outfile", type=str, default="data/instruct_text.txt")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

with open(args.infile, "r", encoding="utf-8") as inf, open(args.outfile, "w", encoding="utf-8") as outf:
    for line in inf:
        j = json.loads(line)
        instr = j.get("instruction", "").strip()
        inp = j.get("input", "").strip()
        out = j.get("output", "").strip()

        # Format: instruction + input + response separated clearly
        if inp:
            prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}\n\n"
        else:
            prompt = f"### Instruction:\n{instr}\n\n### Response:\n{out}\n\n"

        outf.write(prompt)

print("Wrote", args.outfile)
