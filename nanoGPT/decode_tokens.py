import numpy as np
import pickle
import os

DATA_DIR = "data/music_char_nanogpt"

meta_path = os.path.join(DATA_DIR, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

itos = meta["itos"]  # maps token -> event
stoi = meta["stoi"]
vocab_size = meta["vocab_size"]

print("Loaded vocab size =", vocab_size)

def decode(arr):
    """Return raw token events as a list (not characters)."""
    return [itos[int(i)] for i in arr]

BIN_FILE = os.path.join(DATA_DIR, "train.bin")
print("Loading:", BIN_FILE)

data = np.fromfile(BIN_FILE, dtype=np.uint16)
print("Total tokens:", len(data))

start = 0
end = 2000  # safe slice

decoded_events = decode(data[start:end])

print("\n=== DECODED EVENT TOKENS ===")
print(decoded_events[:200])  # print first 200 events
print("=== END ===")
