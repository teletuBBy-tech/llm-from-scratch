# src/infer/generate.py
import argparse
import math
import torch
from src.model.gpt import GPT
from src.data.tokenizer import TiktokenWrapper
import json
import sys

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    # logits: (B, V) tensor
    logits = logits.clone()
    V = logits.size(-1)

    # top-k
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        # topk_vals shape: (B, top_k)
        min_topk = topk_vals[:, -1].unsqueeze(1)  # (B,1)
        logits[logits < min_topk] = -1e10

    # top-p (nucleus)
    if top_p is not None and top_p > 0.0 and top_p < 1.0:
        sorted_vals, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_vals, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        # mask out tokens with cumulative prob above top_p
        sorted_idx_to_remove = cumulative_probs > top_p
        # keep at least one token
        sorted_idx_to_remove[..., 0] = False
        # build mask for original indices
        batch_indices = []
        indices_to_remove = []
        # flattened approach
        for b in range(sorted_idx.size(0)):
            remove_idx = sorted_idx[b, sorted_idx_to_remove[b]]
            logits[b, remove_idx] = -1e10

    return logits

def apply_repetition_penalty(logits, generated_ids, penalty):
    # logits: (1, V) or (B, V)
    if penalty is None or penalty == 1.0 or len(generated_ids) == 0:
        return logits
    # handle batch dim
    for tok in set(generated_ids):
        logits[..., tok] = logits[..., tok] / penalty
    return logits

def generate(model, ids, max_new_tokens=50, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, device="cpu"):
    model.eval()
    generated = ids.tolist()[0].copy()
    input_ids = ids.to(device)

    for _ in range(max_new_tokens):
        # forward
        logits = model(input_ids[:, -model.context_length:])  # (B, T, V)
        logits = logits[:, -1, :] / max(1e-8, temperature)   # (B, V)

        # apply repetition penalty
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)

        # apply top-k / top-p filtering
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B,1)

        # append and extend input_ids
        generated.append(int(next_id.item()))
        input_ids = torch.cat([input_ids, next_id.to(device)], dim=1)

    return torch.tensor([generated], dtype=torch.long)

def resize_pos_emb(checkpoint_pos_emb: torch.Tensor, new_len: int) -> torch.Tensor:
    """
    checkpoint_pos_emb: (old_len, dim)
    returns resized (new_len, dim) by linear interpolation on the length axis
    """
    old_len, dim = checkpoint_pos_emb.shape
    if old_len == new_len:
        return checkpoint_pos_emb
    # transpose to (1, dim, old_len) for interpolate
    p = checkpoint_pos_emb.t().unsqueeze(0)  # (1, dim, old_len)
    p = torch.nn.functional.interpolate(p, size=new_len, mode='linear', align_corners=False)
    p = p.squeeze(0).t()  # (new_len, dim)
    return p

def sanitize_state_for_model(state_dict: dict, model):
    """
    - If keys like 'model_state_dict' are used, caller should unwrap before calling this.
    - Resize pos_emb.weight if necessary.
    - Remove attention masks or buffers that have mismatched shapes.
    - Drop extra transformer block keys (e.g., blocks.4.*) if checkpoint has more layers than model.
    Returns modified state_dict.
    """
    new_state = {}
    model_keys = set(model.state_dict().keys())

    # quick accessors
    # 1) Handle pos emb resizing if present
    # find possible pos emb key
    possible_pos_keys = [k for k in state_dict.keys() if "pos_emb" in k or "pos_embeddings" in k or "pos_embedding" in k]
    for k, v in state_dict.items():
        # if positional embedding present and shape mismatch -> resize
        if k in possible_pos_keys and isinstance(v, torch.Tensor):
            m_tensor = model.state_dict().get(k)
            if m_tensor is not None and v.shape != m_tensor.shape:
                try:
                    if v.ndim == 2 and m_tensor.ndim == 2:
                        print(f"[patch] resizing pos emb '{k}' from {v.shape} -> {m_tensor.shape}")
                        v = resize_pos_emb(v, new_len=m_tensor.shape[0])
                    else:
                        print(f"[patch] incompatible pos emb dims for '{k}', skipping resize (ckpt {v.shape}, model {m_tensor.shape})")
                        # skip; will let load_state_dict handle missing keys
                except Exception as e:
                    print(f"[patch] pos emb resize failed for {k}: {e}", file=sys.stderr)

        # skip attention masks or buffers that don't match
        if k.endswith(".attn.mask") or k.endswith(".attn_mask") or k.endswith(".attn.mask.tensor"):
            # if shape mismatch, skip
            model_mask = None
            if k in model.state_dict():
                model_mask = model.state_dict()[k]
            if model_mask is not None and isinstance(v, torch.Tensor) and v.shape != model_mask.shape:
                print(f"[patch] dropping mismatched mask key '{k}' from checkpoint (ckpt {v.shape} vs model {model_mask.shape})")
                continue
            # else keep it (if shapes match)
        new_state[k] = v

    # drop extra blocks if checkpoint has more layers
    # find checkpoint max block index and model max block index
    ckpt_block_idxs = set()
    model_block_idxs = set()
    import re
    block_re = re.compile(r"blocks\.(\d+)\.")
    for k in new_state.keys():
        m = block_re.search(k)
        if m:
            ckpt_block_idxs.add(int(m.group(1)))
    for k in model.state_dict().keys():
        m = block_re.search(k)
        if m:
            model_block_idxs.add(int(m.group(1)))
    if ckpt_block_idxs and model_block_idxs:
        max_ckpt = max(ckpt_block_idxs)
        max_model = max(model_block_idxs)
        if max_ckpt > max_model:
            # drop keys with block index > max_model
            to_drop = [k for k in new_state.keys() if block_re.search(k) and int(block_re.search(k).group(1)) > max_model]
            for k in to_drop:
                print(f"[patch] dropping checkpoint key for extra block: {k}")
                new_state.pop(k, None)

    return new_state

def load_model(ckpt_path, tok, cfg, device):
    model = GPT(
        vocab_size=tok.vocab_size,
        context_length=cfg["context_length"],
        embed_dim=cfg["embed_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg.get("dropout", 0.1),
    )

    raw = torch.load(ckpt_path, map_location=device)
    # Unwrap common wrappers
    if isinstance(raw, dict) and ("model_state_dict" in raw or "state_dict" in raw):
        if "model_state_dict" in raw:
            state = raw["model_state_dict"]
        else:
            state = raw.get("state_dict", raw)
    elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        state = raw
    else:
        # unknown format
        raise RuntimeError(f"Unrecognized checkpoint format at {ckpt_path}")

    # sanitize and adapt some shapes where safe
    state = sanitize_state_for_model(state, model)

    # Now attempt to load with strict=False and show diagnostics
    model_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    # Try loading
    try:
        res = model.load_state_dict(state, strict=False)
        print("[patch] load_state_dict result:", res)
    except RuntimeError as e:
        # fallback: attempt to remove keys with mismatched shapes
        print("[patch] load_state_dict failed first attempt:", e)
        # filter out keys with mismatched shapes between state and model
        filtered = {}
        for k, v in state.items():
            if k in model_state:
                if isinstance(v, torch.Tensor) and v.shape != model_state[k].shape:
                    print(f"[patch] skipping key due to shape mismatch: {k} ckpt {v.shape} != model {model_state[k].shape}")
                    continue
            filtered[k] = v
        res = model.load_state_dict(filtered, strict=False)
        print("[patch] load_state_dict result after filtering:", res)

    return model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/instruct_train.json",
                        help="path to the training config JSON used to create the checkpoint")
    parser.add_argument("--ckpt", type=str, default="checkpoints/epoch_3.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    print("Device:", args.device)
    print("Loading tokenizer...")
    tok = TiktokenWrapper("gpt2")

    # Load the exact training config so model shape matches checkpoint
    with open(args.config, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    # Build cfg directly from training config (require keys to exist)
    cfg = {
        "context_length": train_cfg["context_length"],
        "embed_dim": train_cfg["embed_dim"],
        "n_layers": train_cfg["n_layers"],
        "n_heads": train_cfg["n_heads"],
        "dropout": train_cfg.get("dropout", 0.1),
    }

    print("Using config:", args.config)
    print("Training cfg (summary):", {k: cfg[k] for k in ["context_length","embed_dim","n_layers","n_heads"]})

    print("Loading model from:", args.ckpt)
    model = load_model(args.ckpt, tok, cfg, args.device)
    print("Model loaded. Generating...")

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long).to(args.device)
    out = generate(model, ids, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                   repetition_penalty=args.repetition_penalty, device=args.device)

    text = tok.decode(out[0].tolist())
    print("\n----- GENERATED TEXT -----\n")
    print(text)
    print("\n----- END -----")
