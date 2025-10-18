#!/usr/bin/env python3
# convert_sep28k_to_wenet.py
# Usage: python convert_sep28k_to_wenet.py sep28k_annotations.csv wav_dir output_data_dir

import csv, os, sys, math
from pathlib import Path

if len(sys.argv) < 4:
    print("Usage: python convert_sep28k_to_wenet.py annotations.csv wav_dir output_data_dir")
    sys.exit(1)

ann_csv = sys.argv[1]
wav_dir = Path(sys.argv[2])
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

wav_scp = out / "wav.scp"
segments = out / "segments"
text = out / "text"

# Modify these names to match columns in your CSV
# Example: columns: clip_id,audio_path,block,prolongation,soundrep,wordrep,interjection
with open(ann_csv, newline='', encoding='utf-8') as f, \
     open(wav_scp, 'w', encoding='utf-8') as fwav, \
     open(segments, 'w', encoding='utf-8') as fseg, \
     open(text, 'w', encoding='utf-8') as ftxt:
    reader = csv.DictReader(f)
    for row in reader:
        clip = row.get('clip_id') or row.get('id') or row.get('uid')
        # find wav path in wav_dir by clip id, or use audio_url if you already downloaded with the same name
        wav_path = wav_dir / f"{clip}.wav"
        if not wav_path.exists():
            # try other naming strategies
            # skip if missing
            print("Missing wav for", clip)
            continue
        rec_id = f"rec_{clip}"
        utt_id = f"utt_{clip}"
        # event columns: adjust keys as necessary
        ev_block = int(row.get('block', '0') or 0)
        ev_prolong = int(row.get('prolongation', row.get('prolong', '0') or 0))
        ev_sound = int(row.get('soundrep', '0') or 0)
        ev_word = int(row.get('wordrep', '0') or 0)
        ev_interj = int(row.get('interjection', '0') or 0)
        is_dys = 1 if (ev_block or ev_prolong or ev_sound or ev_word or ev_interj) else 0
        # write wav.scp (use absolute path)
        fwav.write(f"{rec_id} {wav_path.resolve()}\n")
        # duration: optionally you can compute with soundfile or so; here we use a placeholder 3.0
        duration = 3.0
        fseg.write(f"{utt_id} {rec_id} 0.00 {duration:.2f}\n")
        # text: store labels as 'is_dys ev_block ev_prolong ev_sound ev_word ev_interj'
        label_str = f"{is_dys} {ev_block} {ev_prolong} {ev_sound} {ev_word} {ev_interj}"
        ftxt.write(f"{utt_id} {label_str}\n")

print("Wrote to", out)
