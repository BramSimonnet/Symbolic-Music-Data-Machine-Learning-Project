import os
import subprocess
from multiprocessing import Pool

INPUT_DIR = "data/raw_midi"
OUTPUT_DIR = "data/processed/abc_fast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_one(midi_path):
    try:
        out = os.path.join(
            OUTPUT_DIR,
            os.path.basename(midi_path).replace(".mid", ".abc")
        )

        if os.path.exists(out):
            return f"SKIP {out}"

        # run midi2abc
        cmd = ["midi2abc", midi_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        abc_text = result.stdout

        with open(out, "w") as f:
            f.write(abc_text)

        return f"OK {out}"

    except Exception as e:
        return f"FAIL {midi_path}: {e}"

def walk_midis():
    paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(".mid"):
                paths.append(os.path.join(root, f))
    return paths

if __name__ == "__main__":
    files = walk_midis()
    print("Found", len(files), "MIDI files")

    with Pool(processes=os.cpu_count()) as p:
        for msg in p.imap_unordered(convert_one, files, chunksize=32):
            print(msg)
