import os

input_dir = "data/raw"
output_file = "data/train.txt"

count_files = 0
total_chars = 0

print(f"\n=== MERGING DATASET FILES FROM: {input_dir} ===\n")

with open(output_file, "w", encoding="utf-8") as out:
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".txt"):
            continue  # skip non-text files

        path = os.path.join(input_dir, fname)
        print(f"[+] Merging: {path}")

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"[!] Failed to read {fname}: {e}")
            continue

        if not text or len(text) < 20:  # skip empty or useless files
            print(f"[!] Skipping empty/small file: {fname}")
            continue

        out.write(text + "\n\n")  # separator between files

        count_files += 1
        total_chars += len(text)

print("\n=== MERGE COMPLETE ===")
print(f"Files merged: {count_files}")
print(f"Total characters: {total_chars:,}")
print(f"Merged training file saved to: {output_file}\n")

