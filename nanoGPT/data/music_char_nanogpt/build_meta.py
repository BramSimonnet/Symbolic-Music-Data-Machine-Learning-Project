import os
import pickle
import numpy as np

DATA_DIR = os.path.dirname(__file__)
print("DATA_DIR =", DATA_DIR)

train = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)
val   = np.fromfile(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16)
test  = np.fromfile(os.path.join(DATA_DIR, "test.bin"), dtype=np.uint16)

all_ids = np.concatenate([train, val, test])
vocab = sorted(list(set(all_ids.tolist())))
vocab_size = len(vocab)

print(f"Detected vocab size = {vocab_size}")

stoi = {i: i for i in vocab}
itos = {i: i for i in vocab}

meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}

out_path = os.path.join(DATA_DIR, "meta.pkl")
with open(out_path, "wb") as f:
    pickle.dump(meta, f)

print(f"Saved meta.pkl to: {out_path}")
