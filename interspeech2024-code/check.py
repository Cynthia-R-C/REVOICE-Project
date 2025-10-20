# Cynthia Chen 10/19/2025
# Purpose: check for empty audio files in wav.scp due to error in cmvn calculation script detecting .wav files with 0 samples

from pathlib import Path
import torchaudio

count = 0
for line in open("data/train/wav.scp", encoding="utf8"):
    utt, path = line.strip().split(maxsplit=1)
    try:
        waveform, sr = torchaudio.load(path.strip('"'))
        if waveform.numel() == 0:
            print(f"⚠️ Empty audio: {utt} -> {path}")
            count += 1
    except Exception as e:
        print(f"❌ Error loading {utt}: {path}\n{e}")
        count += 1

print(f"Total problematic files: {count}")
