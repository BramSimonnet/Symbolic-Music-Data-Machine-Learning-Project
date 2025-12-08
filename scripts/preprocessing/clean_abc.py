import os
import re
from multiprocessing import Pool, cpu_count

RAW_DIR = "data/processed/midi_abc"
OUT_DIR = "data/processed/midi_abc_clean"
os.makedirs(OUT_DIR, exist_ok=True)

REMOVE_PATTERNS = [
    r"^calling midi2abc.*",
    r"^mid2abc.*",
    r"^X:.*", r"^T:.*", r"^M:.*", r"^L:.*", r"^Q:.*",
    r"^K:.*", r"^V:.*", r"^%.*", r"^\s*$"
]

compiled_patterns = [re.compile(p) for p in REMOVE_PATTERNS]

def clean_one(file):
    in_path = os.path.join(RAW_DIR, file)
    out_path = os.path.join(OUT_DIR, file)

    try:
        with open(in_path, "r") as f:
            lines = f.readlines()
    except:
        return None

    cleaned = []
    for line in lines:
        line = line.strip()

        if any(p.match(line) for p in compiled_patterns):
            continue

        line = line.replace("\\", "")
        line = line.replace("\t", " ")

        if line:
            cleaned.append(line)

    if not cleaned:
        return None

    final = " ".join(cleaned)
    final = re.sub(r"\s+", " ", final).strip()

    try:
        with open(out_path, "w") as f:
            f.write(final)
    except:
        return None

    return file

def main():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".abc")]
    print(f"Cleaning {len(files)} ABC files using {cpu_count()} cores...")

    with Pool(cpu_count()) as pool:
        for _ in pool.imap_unordered(clean_one, files, chunksize=100):
            pass

if __name__ == "__main__":
    main()
