import random
from pathlib import Path
from pydub import AudioSegment

INPUT_DIR = Path("data/vad_segments")
OUTPUT_DIR = Path("data/single_channels")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for wav_path in INPUT_DIR.rglob("*.wav"):
    relative = wav_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / relative

    if out_path.exists():
        print(f"⏩ Già fatto: {relative}")
        continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(wav_path)

    if audio.channels == 1:
        # Raddoppia il canale mono
        left, right = audio, audio
    else:
        left, right = audio.split_to_mono()

    # Azzera un canale a caso
    if random.choice(["L", "R"]) == "L":
        left = AudioSegment.silent(duration=len(left), frame_rate=left.frame_rate)
    else:
        right = AudioSegment.silent(duration=len(right), frame_rate=right.frame_rate)

    # Uniforma lunghezza
    min_len = min(len(left), len(right))
    left = left[:min_len]
    right = right[:min_len]

    # Ricrea stereo e salva
    stereo = AudioSegment.from_mono_audiosegments(left, right)
    stereo.export(out_path, format="wav")
    print(f"✅ Mascherato: {relative}")
