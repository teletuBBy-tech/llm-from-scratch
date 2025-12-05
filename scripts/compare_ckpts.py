# scripts/compare_ckpts.py
import json, torch, time
from src.data.tokenizer import TiktokenWrapper
from src.model.gpt import GPT
from pathlib import Path

PROMPTS = [
    "Explain recursion in simple terms",
    "How do I write a unit test in Python?",
    "Summarize the book 'The Great Gatsby' in 3 sentences.",
    "Give me 5 tips to improve code readability.",
    "What is dynamic programming, with an example?",
]

def load_model_from(cfg_file, ckpt_path, device="cpu"):
    cfg = json.load(open(cfg_file))
    tok = TiktokenWrapper("gpt2")
    model = GPT(vocab_size=tok.vocab_size, context_length=cfg["context_length"],
                embed_dim=cfg["embed_dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"])
    raw = torch.load(ckpt_path, map_location=device)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, tok, cfg

def generate_text(model, tok, prompt, max_new=120, temp=0.7, top_k=50, top_p=0.9, device="cpu"):
    import torch.nn.functional as F
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(device)
    generated = ids.tolist()[0].copy()
    input_ids = ids
    for _ in range(max_new):
        logits = model(input_ids[:, -model.context_length:])
        logits = logits[:, -1, :]/max(1e-8, temp)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated.append(int(next_id.item()))
        input_ids = torch.cat([input_ids, next_id.to(device)], dim=1)
    return tok.decode(generated)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_cfg = "configs/train_small.json"
    base_ckpt = "checkpoints/epoch_5.pt"
    instr_cfg = "configs/train_instruct.json"
    instr_ckpt = "checkpoints_instruct/epoch_6.pt"

    base_model, base_tok, _ = load_model_from(base_cfg, base_ckpt, device=device)
    instr_model, instr_tok, _ = load_model_from(instr_cfg, instr_ckpt, device=device)

    out = []
    for p in PROMPTS:
        t0 = time.time()
        b = generate_text(base_model, base_tok, p, device=device)
        i = generate_text(instr_model, instr_tok, p, device=device)
        out.append({"prompt": p, "base": b, "instr": i, "time": time.time()-t0})
        print("PROMPT:", p)
        print("BASE:\n", b)
        print("INSTRUCT:\n", i)
        print("-"*60)

    Path("compare_outputs.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Wrote compare_outputs.json")

if __name__ == "__main__":
    main()
