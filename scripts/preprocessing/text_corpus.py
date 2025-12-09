import os

CLEAN_DIR = "data/processed/midi_abc_clean"
OUT_PATH = "data/processed/music.txt"

with open(OUT_PATH, "w") as out:
    for fname in os.listdir(CLEAN_DIR):
        if not fname.endswith(".abc"):
            continue

        path = os.path.join(CLEAN_DIR, fname)
        try:
            with open(path, "r") as f:
                text = f.read().strip()
        except:
            continue

        out.write(text + "\n")

print("DONE. Saved combined text to:", OUT_PATH)
