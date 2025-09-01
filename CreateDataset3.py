import os
import pandas as pd
import subprocess
from pathlib import Path
import hashlib

# === CONFIGURAZIONE ===
EXCEL_PATH = "singfake_icassp.csv"
OUTPUT_DIR = Path("data/my_dataset")
MAX_VIDEOS = None  # es. 100 per test

# === LEGGI CSV ===
df = pd.read_csv(EXCEL_PATH)
print(f"Righe totali trovate: {len(df)}")

# === FILTRA COLONNE IMPORTANTI ===
df = df[["Set", "Bonafide Or Spoof", "Url"]].dropna()
df = df[df["Bonafide Or Spoof"].isin(["bonafide", "spoof"])]

if MAX_VIDEOS:
    df = df.head(MAX_VIDEOS)

# === FUNZIONE PER CREARE NOME FILE UNIVOCO ===
def url_to_filename(url):
    return hashlib.md5(url.encode()).hexdigest() + ".wav"

# === GRUPPA PER SET ===
sets = df["Set"].unique()

for set_name in sets:
    print(f"\n📁 Processing set: {set_name}")
    df_set = df[df["Set"] == set_name]

    out_dir = OUTPUT_DIR / set_name
    bonafide_dir = out_dir / "bonafide"
    spoof_dir = out_dir / "spoof"
    protocol_path = out_dir / "protocol.csv"

    # Crea cartelle
    bonafide_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    protocol_lines = []

    for idx, row in enumerate(df_set.itertuples(), start=1):
        label = row._2.lower()  # Bonafide Or Spoof
        url = row.Url
        file_id = url_to_filename(url)
        subdir = bonafide_dir if label == "bonafide" else spoof_dir
        wav_path = subdir / file_id

        print(f"[{file_id}] Scarico da: {url}")

        if wav_path.exists():
            print(f"⏩ Già scaricato: {wav_path.relative_to(out_dir)}")
            protocol_lines.append(f"{wav_path.relative_to(out_dir)},{label}")
            continue

        try:
            subprocess.run([
                "yt-dlp",
                "--no-playlist",
                "-f", "bestaudio",
                "--extract-audio",
                "--audio-format", "wav",
                "--output", str(wav_path),
                url
            ], check=True)

            protocol_lines.append(f"{wav_path.relative_to(out_dir)},{label}")
        except subprocess.CalledProcessError:
            print(f"❌ Errore con: {url}")

    # Scrivi protocollo
    with open(protocol_path, "w") as f:
        f.write("filename,label\n")
        f.write("\n".join(protocol_lines))

    print(f"✅ {set_name} completato: {len(protocol_lines)} file salvati.")
