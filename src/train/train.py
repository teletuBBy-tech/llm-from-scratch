import os
import json
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from src.model.gpt import GPT
from src.data.dataset import create_dataloader_v1

def calc_loss(model, xb, yb, device):
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)                      # expects (B, T, V)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        yb.view(-1)
    )
    return loss

def train(cfg):
    # make dirs
    os.makedirs(cfg.get("ckpt_dir", "checkpoints"), exist_ok=True)
    os.makedirs(cfg.get("log_dir", "runs/exp"), exist_ok=True)

    # read raw text (if dataloader expects text input)
    with open(cfg["data_file"], "r", encoding="utf-8") as f:
        text = f.read()

    # dataloader + tokenizer
    dl, tok = create_dataloader_v1(
        text,
        batch_size=cfg["batch_size"],
        max_length=cfg["context_length"],
        stride=cfg["stride"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPT(
        vocab_size=tok.vocab_size,
        context_length=cfg["context_length"],
        embed_dim=cfg["embed_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"]
    ).to(device)

    print("Model params:", sum(p.numel() for p in model.parameters()))
    optim = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    writer = SummaryWriter(cfg["log_dir"])

    use_amp = cfg.get("use_amp", False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl:
            # forward + backward (AMP-aware)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = calc_loss(model, xb, yb, device)

            scaler.scale(loss).backward()

            # optional gradient clipping
            max_norm = cfg.get("grad_clip_norm", None)
            if max_norm is not None:
                scaler.unscale_(optim)  # unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optim)
            scaler.update()

            epoch_loss += loss.item()
            if global_step % cfg["log_every"] == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)

            if global_step % cfg.get("print_every", 50) == 0:
                print(f"[E{epoch+1}] step {global_step} loss={loss.item():.4f}")

            global_step += 1

        avg_loss = epoch_loss / (len(dl) if len(dl) > 0 else 1)
        print(f"Epoch {epoch+1}/{cfg['epochs']} - avg_loss={avg_loss:.4f}")

        # save checkpoint (epoch-level)
        ckpt_path = os.path.join(cfg["ckpt_dir"], f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)

    writer.close()
    print("Training finished. Last checkpoint:", ckpt_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_small.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train(cfg)
