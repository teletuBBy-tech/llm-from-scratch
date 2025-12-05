# scripts/count_instruct_markers.py
from pathlib import Path

p = Path("data/instruct_text.txt")

if p.exists():
    text = p.read_text(encoding="utf-8", errors="ignore")
else:
    print("File not found:", p)
    text = ""

markers = [
    "### Instruction:",
    "### Response:",
    "### Input:",
    "\"instruction\"",
    "\"response\"",
    "Instruction:",
    "Response:"
]

# Count occurrences
print("Marker counts:")
for m in markers:
    count = text.count(m)
    print(f"{m}: {count}")

# Show sample context around first occurrence of each marker
for m in markers:
    idx = text.find(m)
    if idx != -1:
        print(f"\n--- Sample around '{m}' ---")
        snippet = text[max(0, idx - 200) : idx + 200]
        print(snippet.replace("\n", "␤"))

