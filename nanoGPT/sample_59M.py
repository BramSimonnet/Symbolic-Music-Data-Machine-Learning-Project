import os
import json
import torch
from model import GPTConfig, GPT

OUT_DIR = "transformer-59M-finalpart4"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")
VOCAB_PATH = "../data/processed/vocab.json"

SAMPLES_DIR = "generated_abc2"
os.makedirs(SAMPLES_DIR, exist_ok=True)

NUM_UNCONDITIONAL = 5
NUM_CONDITIONAL = 5
MAX_NEW_TOKENS = 400
TEMPERATURE = 1.0
TOP_K = 50

with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

itos = {int(k): v for k, v in vocab["itos"].items()}
stoi = vocab["stoi"]
vocab_size = len(itos)

def encode(c):
    return stoi[c] if c in stoi else stoi[" "]

def decode(i):
    return itos[i]

checkpoint = torch.load(CKPT_PATH, map_location="cpu")
cfg = checkpoint["config"]

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=cfg["block_size"],
    n_layer=cfg["n_layer"],
    n_head=cfg["n_head"],
    n_embd=cfg["n_embd"],
    dropout=0.0,
)

model = GPT(config)
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()

def generate(prompt="", max_new_tokens=300, temperature=1.0, top_k=50):
    if prompt == "":
        x = torch.zeros((1, 1), dtype=torch.long)
    else:
        x = torch.tensor([encode(c) for c in prompt], dtype=torch.long)[None, :]

    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature, top_k)

    return "".join(decode(int(i)) for i in y[0])

CONDITIONAL_PROMPTS = [
    "X:1\nT:Sample\nM:4/4\nK:C\n",
    "X:1\nT:Sample\nM:3/4\nK:G\n",
    "X:1\nT:Sample\nM:6/8\nK:D\n",
    "X:1\nT:Sample\nM:4/4\nK:Am\n",
    "X:1\nT:Sample\nM:2/4\nK:F\n",
]

for i in range(NUM_UNCONDITIONAL):
    abc = generate(
        prompt="",
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )
    path = os.path.join(SAMPLES_DIR, f"unconditional_{i+1}.abc")
    with open(path, "w") as f:
        f.write(abc)
    print(f"Saved {path}")

for i, prompt in enumerate(CONDITIONAL_PROMPTS[:NUM_CONDITIONAL]):
    abc = generate(
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )
    path = os.path.join(SAMPLES_DIR, f"conditional_{i+1}.abc")
    with open(path, "w") as f:
        f.write(abc)
    print(f"Saved {path}")

print("\nABC files saved to:", SAMPLES_DIR)
