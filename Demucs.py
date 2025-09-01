import os
import random
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy.signal import resample_poly
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile
import torch

# === CONFIG ===
ROOT_DIR = Path("data/my_dataset")
OUTPUT_DIR = Path("data/processed_demucs")
EXCLUDE_DIRS = ["extra"]
SAMPLE_RATE = 16000
USE_CUDA = torch.cuda.is_available()

# === DEMUCS MODEL ===
print("🔁 Carico Demucs (pretrained)...")
model = get_model(name="htdemucs")
if USE_CUDA:
    model.to("cuda")  # Sposta il modello sulla GPU
print(f"🚀 Uso {'GPU' if USE_CUDA else 'CPU'}")

# === FUNZIONE: processa un file ===
def process_with_demucs(wav_path: Path, out_path: Path):
    if out_path.exists():
        print(f"⏩ Già elaborato: {wav_path.relative_to(ROOT_DIR)}")
        return
    try:
        audio = AudioFile(str(wav_path))
        wav = audio.read(streams=0, samplerate=model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        # === Sposta anche il segnale sulla GPU se disponibile
        wav_tensor = wav[None].detach().clone().float()
        if USE_CUDA:
            wav_tensor = wav_tensor.to("cuda")

        sources = apply_model(model, wav_tensor, split=True, overlap=0.25)[0]

        vocals = sources[model.sources.index("vocals")]
        vocals = vocals[random.choice([0, 1])].cpu().numpy()  # scegli canale e torna su CPU

        if model.samplerate != SAMPLE_RATE:
            vocals = resample_poly(vocals, SAMPLE_RATE, model.samplerate)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), vocals, SAMPLE_RATE)
        print(f"✅ {wav_path.relative_to(ROOT_DIR)} → {out_path.relative_to(OUTPUT_DIR)}")

    except Exception as e:
        print(f"❌ Errore con {wav_path.name}: {e}")

# === RACCOLTA FILE ===
all_wavs = [wav for wav in ROOT_DIR.rglob("*.wav")
            if not any(str(wav.relative_to(ROOT_DIR)).startswith(ex) for ex in EXCLUDE_DIRS)]

print(f"🔍 Trovati {len(all_wavs)} file WAV da processare.")

# === LOOP ===
for wav in all_wavs:
    relative = wav.relative_to(ROOT_DIR)
    out_wav = OUTPUT_DIR / relative
    process_with_demucs(wav, out_wav)
