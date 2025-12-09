import numpy as np
import os

TOKENS_PATH = "data/processed/music_tokens.npy"
OUT_DIR = "data/splits"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading tokens...")
tokens = np.load(TOKENS_PATH)

N = len(tokens)
print("Total tokens:", N)

n_train = int(N * 0.98)
n_val = int(N * 0.01)
n_test = N - n_train - n_val

train_tokens = tokens[:n_train]
val_tokens = tokens[n_train:n_train+n_val]
test_tokens = tokens[n_train+n_val:]

print("Saving train/val/test splits...")

np.save(os.path.join(OUT_DIR, "train.npy"), train_tokens)
np.save(os.path.join(OUT_DIR, "val.npy"), val_tokens)
np.save(os.path.join(OUT_DIR, "test.npy"), test_tokens)

print("Done.")
print("Train:", train_tokens.shape)
print("Val:", val_tokens.shape)
print("Test:", test_tokens.shape)
