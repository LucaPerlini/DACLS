#spettrogrammi 128x128 da clip audio, le clip sono suddivise in 1,28s e gli
#spettrogrammi sono con frame size di 20ms e hop di 10ms (quindi overlap 50%)

import argparse
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display

def argparser():
    p = argparse.ArgumentParser(description="Genera log‑spectrogram 128x128 da clip di 1.28s")
    p.add_argument('-i', '--input', required=True, help="Cartella radice con audio")
    p.add_argument('-o', '--output', required=False, help="Dove salvare PNG (opzionale)")
    return p

def create_log_spectrograms(input_root, output_root=None):
    if output_root is None:
        output_root = os.getcwd()

    sr = 16000
    n_fft = int(0.020 * sr)         # 20 ms = 320
    hop_length = int(0.010 * sr)    # 10 ms = 160
    target_duration = 1.28
    target_samples = int(target_duration * sr)  # 20480 samples

    traccia_count = 1  # Contatore globale delle tracce

    for root, _, files in os.walk(input_root):
        rel = os.path.relpath(root, input_root)
        out_dir = os.path.join(output_root, rel)
        os.makedirs(out_dir, exist_ok=True)

        for fn in sorted(files):
            if not fn.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                continue

            in_fp = os.path.join(root, fn)

            try:
                y, _ = librosa.load(in_fp, sr=sr, mono=True)
                total_samples = len(y)

                if total_samples < target_samples:
                    print(f"🗑️ Rimosso clip troppo corto ({total_samples/sr:.2f}s): {os.path.join(rel, fn)}")
                    os.remove(in_fp)
                    continue

                num_clips = total_samples // target_samples

                for i in range(num_clips):
                    y_clip = y[i * target_samples : (i + 1) * target_samples]

                    D = librosa.stft(y_clip, n_fft=n_fft, hop_length=hop_length)
                    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

                    D_db = D_db[:128, :128]

                    # Costruisce il nome tracciaN_segmentoM
                    clip_name = f"traccia{traccia_count}_segmento{i+1}.png"
                    out_fp = os.path.join(out_dir, clip_name)

                    if os.path.exists(out_fp):
                        print(f"⏩ Skipping: {clip_name}")
                        continue

                    plt.figure(figsize=(4, 4))
                    plt.axis('off')
                    plt.axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
                    librosa.display.specshow(D_db, sr=sr, hop_length=hop_length, cmap='gray_r')

                    plt.savefig(out_fp, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print(f"✅ Saved: {os.path.join(rel, clip_name)}")

                traccia_count += 1  # Passa alla prossima traccia

            except Exception as e:
                print(f"❌ Error {os.path.join(rel, fn)}: {e}")

if __name__ == "__main__":
    args = argparser().parse_args()
    create_log_spectrograms(args.input, args.output)
