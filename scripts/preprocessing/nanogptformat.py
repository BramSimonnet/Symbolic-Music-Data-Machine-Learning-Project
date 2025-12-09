import os
import json
import pickle
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[2]
splits_dir = root / "data" / "splits"
processed_dir = root / "data" / "processed"
out_dir = root / "data" / "music_char_nanogpt"
out_dir.mkdir(parents=True, exist_ok=True)

train_tokens = np.load(splits_dir / "train.npy", mmap_mode="r")
val_tokens   = np.load(splits_dir / "val.npy",   mmap_mode="r")
test_tokens  = np.load(splits_dir / "test.npy",  mmap_mode="r")

def save_bin(a, name):
    a.astype(np.uint16).tofile(out_dir / name)

save_bin(train_tokens, "train.bin")
save_bin(val_tokens,   "val.bin")
save_bin(test_tokens,  "test.bin")

with open(processed_dir / "vocab.json", "r") as f:
    vocab_obj = json.load(f)

if isinstance(vocab_obj, dict) and "stoi" in vocab_obj and "itos" in vocab_obj:
    stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in vocab_obj["stoi"].items()}
    itos = {int(k): v for k, v in vocab_obj["itos"].items()}
elif isinstance(vocab_obj, dict) and "vocab" in vocab_obj:
    chars = vocab_obj["vocab"]
    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in itos.items()}
elif isinstance(vocab_obj, list):
    chars = vocab_obj
    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in itos.items()}
else:
    raise ValueError("Bad vocab format")

meta = {"vocab_size": len(stoi), "stoi": stoi, "itos": itos}

with open(out_dir / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("DONE. Saved to:", out_dir)
