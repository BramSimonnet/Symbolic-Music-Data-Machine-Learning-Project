from music21 import converter
import glob

files = glob.glob("generated_abc2/*.abc")

valid = 0
for f in files:
    try:
        converter.parse(f)
        valid += 1
    except Exception:
        pass

print(f"Syntactically valid: {valid}/{len(files)}")
