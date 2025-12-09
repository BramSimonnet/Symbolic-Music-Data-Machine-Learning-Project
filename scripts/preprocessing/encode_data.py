import json
import numpy as np

TEXT_PATH = "data/processed/music.txt"
VOCAB_PATH = "data/processed/vocab.json"
OUT_PATH = "data/processed/music_tokens.npy"

print("Loading vocab...")
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

stoi = vocab["stoi"]

print("Encoding dataset...")
with open(TEXT_PATH, "r") as f:
    data = f.read()

tokens = np.array([stoi[ch] for ch in data], dtype=np.uint16)

print("Saving tokens to:", OUT_PATH)
np.save(OUT_PATH, tokens)

print("Done. Token count:", tokens.shape[0])
