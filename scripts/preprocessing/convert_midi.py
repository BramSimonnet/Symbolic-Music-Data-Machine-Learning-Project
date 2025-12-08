import os
from mido import MidiFile
from tqdm import tqdm

RAW_DIR = "data/raw_midi/lmd_full"
OUT_DIR = "data/processed/midi_txt"

os.makedirs(OUT_DIR, exist_ok=True)

def convert_midi(path):
    try:
        midi = MidiFile(path)
        events = []
        for i, track in enumerate(midi.tracks):
            for msg in track:
                events.append(str(msg))
        return "\n".join(events)
    except Exception as e:
        return None

def main():
    all_files = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(".mid"):
                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} MIDI files")

    for midi_path in tqdm(all_files):
        txt = convert_midi(midi_path)
        if txt is None:
            continue

        out_name = midi_path.replace("/", "_").replace(".mid", ".txt")
        out_path = os.path.join(OUT_DIR, out_name)

        with open(out_path, "w") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
