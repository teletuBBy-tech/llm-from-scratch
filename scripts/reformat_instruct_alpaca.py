# scripts/reformat_instruct_alpaca.py
import re
from pathlib import Path

IN = Path("data/instruct_text.txt")
OUT = Path("data/instruct_text_clean.txt")
BACKUP = Path("data/instruct_text_backup.txt")

if not IN.exists():
    print("Input file not found:", IN)
    raise SystemExit(1)

# Backup original once
if not BACKUP.exists():
    print("Backing up original to", BACKUP)
    BACKUP.write_bytes(IN.read_bytes())

text = IN.read_text(encoding="utf-8", errors="ignore")

# Normalize line endings
text = text.replace("\r\n", "\n").replace("\r", "\n")

# Split big file into blocks separated by at least one blank line or pattern "### Instruction:"
# We'll scan sequentially for markers to be robust.
pattern = re.compile(r"(?:### Instruction:)(.*?)((?=### Instruction:)|\Z)", re.S)

pairs = []
for m in pattern.finditer(text):
    block = m.group(1).strip()
    # within block, find Input (optional) and Response
    # Try to extract Instruction text at block start until Input or Response
    instr = ""
    input_text = ""
    response = ""

    # Find Response marker in block
    resp_m = re.search(r"### Response:(.*)$", block, re.S)
    if resp_m:
        response = resp_m.group(1).strip()
        before_resp = block[:resp_m.start()].strip()
    else:
        # If no response marker, skip
        continue

    # Look for Input marker in before_resp
    in_m = re.search(r"### Input:(.*)$", before_resp, re.S)
    if in_m:
        input_text = in_m.group(1).strip()
        instr = before_resp[:in_m.start()].strip()
    else:
        instr = before_resp.strip()

    # Clean texts: collapse multiple newlines, trim
    def clean(s):
        s = s.strip()
        # collapse consecutive newlines into single newline
        s = re.sub(r"\n{2,}", "\n", s)
        # collapse spaces
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s

    instr_c = clean(instr)
    input_c = clean(input_text)
    resp_c = clean(response)

    if not resp_c or not instr_c:
        # skip incomplete pairs
        continue

    # Build final prompt: instruction + optional input (separated by two newlines)
    if input_c:
        prompt = instr_c + "\n\n" + "Input: " + input_c
    else:
        prompt = instr_c

    pairs.append((prompt, resp_c))

print(f"Found {len(pairs)} instruction-response pairs.")

# Write cleaned file, one pair per line separated by tab (prompt \t response)
with OUT.open("w", encoding="utf-8") as g:
    for p, r in pairs:
        # replace newlines in prompt/response with spaces to keep tokenization consistent for now
        # (You may opt to keep newlines if your dataloader supports it)
        P = p.replace("\n", " ").strip()
        R = r.replace("\n", " ").strip()
        g.write(P + "\t" + R + "\n")

print("Wrote cleaned pairs to", OUT)
