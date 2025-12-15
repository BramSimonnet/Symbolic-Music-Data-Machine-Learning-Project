import os
import argparse
import json
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--vocab_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with open(args.vocab_file, "r") as f:
    vocab = json.load(f)

stoi = vocab["stoi"]
itos = vocab["itos"]
vocab_size = len(stoi)

train_path = os.path.join(args.output_dir, "train.bin")
val_path = os.path.join(args.output_dir, "val.bin")
test_path = os.path.join(args.output_dir, "test.bin")

train_f = open(train_path, "wb")
val_f = open(val_path, "wb")
test_f = open(test_path, "wb")

import os

file_size = os.path.getsize(args.input_file)
train_cutoff = int(file_size * 0.90)
val_cutoff = int(file_size * 0.95)

chunk_size = 1024 * 1024 

pos = 0
with open(args.input_file, "r") as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break

        token_chunk = np.array([stoi[ch] for ch in chunk], dtype=np.uint16)

        if pos < train_cutoff:
            remaining_train = train_cutoff - pos
            if len(chunk) <= remaining_train:
                token_chunk.tofile(train_f)
            else:
                split1 = remaining_train
                train_part = token_chunk[:split1]
                train_part.tofile(train_f)

                token_chunk = token_chunk[split1:]
                pos += split1
                remaining_chunk = token_chunk
    
                token_chunk = remaining_chunk
        if pos >= train_cutoff and pos < val_cutoff:
            remaining_val = val_cutoff - pos
            if len(token_chunk) <= remaining_val:
                token_chunk.tofile(val_f)
            else:
                split2 = remaining_val
                val_part = token_chunk[:split2]
                val_part.tofile(val_f)
                token_chunk = token_chunk[split2:]
                pos += split2
        if pos >= val_cutoff:
            token_chunk.tofile(test_f)

        pos += len(chunk)

train_f.close()
val_f.close()
test_f.close()

meta = {
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
}

with open(os.path.join(args.output_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("DONE. Saved train.bin, val.bin, test.bin, and meta.pkl.")
