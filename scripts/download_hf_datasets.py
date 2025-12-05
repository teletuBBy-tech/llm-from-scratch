# scripts/download_hf_datasets.py
# scripts/download_hf_datasets.py  (patched, robust)
import argparse
import os
from datasets import load_dataset

def save_textfile(dataset, key, out_path, max_items=0):
    print(f"[INFO] Writing → {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            # prefer the given key, else pick first string field
            text = None
            if key in row:
                text = row[key]
            else:
                # fallback: pick first string-like field
                for v in row.values():
                    if isinstance(v, str):
                        text = v
                        break
            if text:
                f.write(text.strip() + "\n\n")
            if max_items > 0 and i + 1 >= max_items:
                break
    print(f"[OK] Saved {min(i+1, max_items) if max_items>0 else (i+1)} items → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/raw")
    parser.add_argument("--max_openweb_examples", type=int, default=0)
    parser.add_argument("--max_wiki_examples", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Salesforce/wikitext
    print("\n=== Downloading WikiText-103 (Salesforce/wikitext) ===")
    ds_wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    save_textfile(ds_wiki, key="text", out_path=os.path.join(args.outdir, "wikitext.txt"), max_items=args.max_wiki_examples)

    # 2) OpenWebText - try multiple fallbacks, with safe sampling
    print("\n=== Downloading OpenWebText (try fallbacks) ===")
    fallback_ids = [
        "Skylion007/openwebtext",  # original you tried (may use dataset script)
        "openwebtext",             # common mirror id (try this)
        "openwebtext/openwebtext", # another possible namespace
        "EleutherAI/openwebtext"   # hypothetical mirror
    ]

    out_path = os.path.join(args.outdir, "openwebtext.txt")
    loaded = False
    for ds_id in fallback_ids:
        try:
            print(f"[TRY] load_dataset('{ds_id}', split='train') ...")
            ds_owt = load_dataset(ds_id, split="train")
            save_textfile(ds_owt, key="text", out_path=out_path, max_items=args.max_openweb_examples)
            loaded = True
            break
        except Exception as e:
            print(f"[WARN] load_dataset('{ds_id}') failed: {e}")
            continue

    if not loaded:
        print("[WARN] Could not load an OpenWebText mirror with `datasets`. Skipping OpenWebText.")
        print("If you really want the full Skylion007/openwebtext, we can use `huggingface_hub.hf_hub_download` in a second step.")
    else:
        print("[OK] OpenWebText saved to", out_path)

    print("\n[DONE] downloads (wiki + openwebtext fallback) completed.\n")
