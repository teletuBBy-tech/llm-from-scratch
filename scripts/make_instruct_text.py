# scripts/make_instruct_text.py
# scripts/make_instruct_text.py  (robust version)
import json, os, codecs

inp = "data/instruct_train.jsonl"
out = "data/instruct_text.txt"
os.makedirs(os.path.dirname(out), exist_ok=True)

good = 0
bad = 0
parts = []
with codecs.open(inp, "r", "utf-8") as f:
    for i, raw in enumerate(f, start=1):
        s = raw.strip()
        if not s:
            continue
        try:
            j = json.loads(s)
            instr = j.get("instruction","").strip()
            input_txt = j.get("input","").strip()
            output = j.get("output","").strip()
            part = f"### Instruction:\n{instr}\n### Input:\n{input_txt}\n### Response:\n{output}\n\n"
            parts.append(part)
            good += 1
        except Exception as e:
            bad += 1
            print(f"[WARN] skipping invalid JSON at line {i}: {e}")
            print("  preview:", repr(s[:200]))
            # continue to next line

with codecs.open(out, "w", "utf-8") as f:
    f.write("".join(parts))

print(f"WROTE {out}  (good={good}, skipped={bad})")

