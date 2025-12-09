import json

TEXT_PATH = "data/processed/music.txt"
VOCAB_PATH = "data/processed/vocab.json"

with open(TEXT_PATH, "r") as f:
    data = f.read()

chars = sorted(list(set(data)))

print("Found", len(chars), "unique characters:")
print(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

with open(VOCAB_PATH, "w") as f:
    json.dump({"vocab": chars, "stoi": stoi, "itos": itos}, f)

print("Saved vocab to:", VOCAB_PATH)
