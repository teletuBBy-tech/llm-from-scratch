# src/train/eval_val.py
import argparse, json, os
import math
import torch
from torch.utils.data import DataLoader
from src.model.gpt import GPT
from src.data.dataset import create_dataloader_v1
from src.data.tokenizer import TiktokenWrapper

def calc_loss(model, xb, yb, device, _warn=[False]):
    xb, yb = xb.to(device), yb.to(device)    # (B, T)
    logits = model(xb)                       # (B, T_model, V)

    # align time dimension in case of off-by-one/misaligned samples
    T_logits = logits.size(1)
    T_targets = yb.size(1)
    if T_logits != T_targets:
        if not _warn[0]:
            print(f"WARNING: time-dim mismatch logits T={T_logits} vs targets T={T_targets}. Trimming to min.")
            _warn[0] = True
        T = min(T_logits, T_targets)
        logits = logits[:, :T, :]
        yb = yb[:, :T]

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        yb.view(-1)
    )
    return loss.item()


def evaluate(ckpt_path, val_file, cfg, batch_size=4, max_batches=None, device='cpu'):
    print("Device:", device)
    tok = TiktokenWrapper("gpt2")
    with open(val_file, "r", encoding="utf-8") as f:
        text = f.read()
    dl, _ = create_dataloader_v1(text, batch_size=batch_size, max_length=cfg["context_length"], stride=cfg["stride"], shuffle=False)

    model = GPT(
        vocab_size=tok.vocab_size,
        context_length=cfg["context_length"],
        embed_dim=cfg["embed_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg.get("dropout", 0.1)
    )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, (xb,yb) in enumerate(dl):
            if max_batches is not None and i >= max_batches:
                break
            loss = calc_loss(model, xb, yb, device)
            total_loss += loss
            n += 1
    avg_loss = total_loss / (n or 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float('inf')  # avoid overflow
    print(f"Evaluated {n} batches. avg_val_loss={avg_loss:.4f}, perplexity={ppl:.4f}")
    return avg_loss, ppl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--val", required=True, help="validation text file (raw text)")
    parser.add_argument("--config", default="configs/train_small.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    evaluate(args.ckpt, args.val, cfg, batch_size=args.batch_size, max_batches=args.max_batches, device=args.device)

