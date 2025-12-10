import os
import time
import math
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_size",
    type=str,
    default="tiny",
    choices=["tiny", "small", "medium", "large"],
    help="Which LSTM size to train.",
)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--max_iters", type=int, default=2500)
parser.add_argument("--eval_interval", type=int, default=250)
parser.add_argument("--lr", type=float, default=3e-4)
args = parser.parse_args()

device = args.device

MODEL_CONFIGS = {
    "tiny": { 
        "d_model": 256,
        "num_layers": 2,
    },
    "small": { 
        "d_model": 320,
        "num_layers": 6,
    },
    "medium": { 
        "d_model": 704,
        "num_layers": 5,
    },
    "large": { 
        "d_model": 1024,
        "num_layers": 6,
    },
}

cfg = MODEL_CONFIGS[args.model_size]

data_dir = "data/music_char_nanogpt"

with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]

train_data = np.memmap(
    os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
)
val_data = np.memmap(
    os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
)

batch_size = args.batch_size
block_size = args.block_size


def get_batch(split):
    """Sample a batch of sequences from train/val."""
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)

    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    x = torch.from_numpy(x.astype(np.int64)).to(device)
    y = torch.from_numpy(y.astype(np.int64)).to(device)
    return x, y

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, hidden=None):
        x = self.emb(idx)
        x, hidden = self.lstm(x, hidden)
        x = self.ln(x)
        logits = self.head(x) 

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )

        return logits, loss, hidden


model = LSTMLM(
    vocab_size=vocab_size,
    d_model=cfg["d_model"],
    num_layers=cfg["num_layers"],
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"number of parameters: {n_params/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


@torch.no_grad()
def estimate_loss(eval_batches=200):
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    for split in ["train", "val"]:
        out_losses = []
        for _ in range(eval_batches):
            xb, yb = get_batch(split)
            _, loss, _ = model(xb, yb)
            out_losses.append(loss.item())
        losses[split] = float(np.mean(out_losses))
    model.train()
    return losses

max_iters = args.max_iters
eval_interval = args.eval_interval

t0 = time.time()
prev_t = t0

for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss(eval_batches=50)
        train_loss = losses["train"]
        val_loss = losses["val"]
        elapsed = time.time() - t0
        print(
            f"step {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
        )

    xb, yb = get_batch("train")
    t1 = time.time()
    _, loss, _ = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    t2 = time.time()

    ms = (t2 - t1) * 1000.0

    print(f"iter {it}: loss {loss.item():.4f}, time {ms:.2f}ms")

total_time = time.time() - t0
print(f"Total training time: {total_time:.1f} seconds")