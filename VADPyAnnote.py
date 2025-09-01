import os
from pathlib import Path
from pyannote.audio import Pipeline
from pydub import AudioSegment

# === CONFIG ===
HF_TOKEN = "hf_your_huggingface_token_here"  # Sostituisci con il tuo token Hugging Face
INPUT_DIR = Path("data/processed_demucs")           # File input
OUTPUT_DIR = Path("data/vad_segments")              # File output
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD PIPELINE ===
print("🔁 Carico pipeline VAD da pyannote...")
vad_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=HF_TOKEN
)

# === LOOP ===
for wav_path in INPUT_DIR.rglob("*.wav"):
    relative = wav_path.relative_to(INPUT_DIR)
    out_dir = OUTPUT_DIR / relative.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Rileva i segmenti vocali
        vad_result = vad_pipeline(wav_path)

        # 2. Carica l'audio completo
        audio = AudioSegment.from_file(wav_path)

        segments = list(vad_result.itersegments())
        if not segments:
            print(f"⚠️ Nessuna voce trovata: {relative}")
            continue

        # 3. Per ogni segmento parlato, salva un file separato
        for idx, segment in enumerate(segments, start=1):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            clip = audio[start_ms:end_ms]

            out_path = out_dir / f"{wav_path.stem}_segment{idx}.wav"
            clip.export(out_path, format="wav")

        print(f"✅ VAD segmentato su: {relative}, segmenti creati: {len(segments)}")

    except Exception as e:
        print(f"❌ Errore su {relative}: {type(e).__name__}: {e}")
