import numpy as np
import os
import pickle

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/music_char_nanogpt")
)


meta_path = os.path.join(DATA_DIR, "meta.pkl")

if not os.path.exists(meta_path):
    print("meta.pkl not found — rebuilding vocabulary metadata...")

    train = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)

    unique_tokens = sorted(list(set(train.tolist())))
    vocab_size = len(unique_tokens)
    itos = {i: i for i in unique_tokens}
    stoi = {i: i for i in unique_tokens}

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }

    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"Rebuilt meta.pkl with vocab_size={vocab_size}")

else:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    vocab_size = meta["vocab_size"]
    itos = meta["itos"]
    stoi = meta["stoi"]

train_data = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)
val_data   = np.fromfile(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16)
test_data  = np.fromfile(os.path.join(DATA_DIR, "test.bin"), dtype=np.uint16)

print(f"Loaded music_char_nanogpt dataset: {len(train_data)=}, {len(val_data)=}, {len(test_data)=}")
print(f"Vocab size = {vocab_size}")
