import numpy as np
import pickle
import os


DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/music_char_nanogpt")
)



with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

vocab_size = meta["vocab_size"]
itos = meta["itos"]
stoi = meta["stoi"]

train_data = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)
val_data = np.fromfile(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16)
test_data = np.fromfile(os.path.join(DATA_DIR, "test.bin"), dtype=np.uint16)
