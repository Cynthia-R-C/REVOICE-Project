# V1
# Cynthia Chen 10/17/2025
# Discarded - this version uses the episode CSV files; this is unnecessarily difficult and incomplete

#!/usr/bin/env python3
# convert_sep28k_to_wenet.py
# Usage: python convert_sep28k_to_wenet.py; assumes path of relevant CSV files; not applicable to other datasets or paths

import csv, os, sys, math
from pathlib import Path

# ann_csv = sys.argv[1]
# wav_dir = Path(sys.argv[2])
# out = Path(sys.argv[3])
# out.mkdir(parents=True, exist_ok=True)

# Paths to the csv files
f_ep_csv = Path('../../raw_data/SEP-28k/fluencybank_episodes.csv')
sep_ep_csv = Path('../../raw_data/SEP-28k/SEP-28k_episodes.csv')
f_label_csv = Path('../../raw_data/SEP-28k/fluencybank_labels.csv')
sep_label_csv = Path('../../raw_data/SEP-28k/SEP-28k_labels.csv')

# Path to audio clips
clip_path = Path('../../raw_data/SEP-28k/clips/stuttering-clips/clips')

# Creating the output paths
out = Path('../data/train_set')
out.mkdir(parents=True, exist_ok=True)  # create the directory if not exist

wav_scp = out / "wav.scp"
segments = out / "segments"
text = out / "text"


# Writes all the correct info for the dataset into the relevant files;
# Run twice, once for FluencyBank CSVs and one for SEP-28k CSVs

def convert_dataset(ep_csv, label_csv):
    '''Convert a given dataset with its episode and label CSV files into WeNet format for StutterNet training.'''

    # Read episode CSV to get wav directory
    with open(ep_csv, newline='', encoding='utf-8') as fep, \
        open(label_csv, newline='', encoding='utf-8') as flabel, \
        open(wav_scp, 'w', encoding='utf-8') as fwav, \
        open(segments, 'w', encoding='utf-8') as fseg, \
        open(text, 'w', encoding='utf-8') as ftxt:

        ep_reader = csv.DictReader(fep)
        label_reader = csv.DictReader(flabel)

        for row in ep_reader:
            pod_name2 = row.get('PodName2')
            ep_ID = row.get('EpID')

            # Formula for audio file path: PodName2_EpID_n
            n = 0  # start with n = 0
            file_ID = f"{pod_name2}_{ep_ID}_{n}"
            wav_dir = clip_path / file_ID

            # Increment n until no more files
            while wav_dir.exists():

                # Write file_ID to fwav
                fwav.write(f"rec_{file_ID} {wav_dir.resolve()}\n")

                # Write file_ID to segments
                fseg.write(f"utt_{file_ID} rec_{file_ID} 0.00 3.00\n")  # placeholder duration 3.00

                # Increment n and update file_ID and wav_dir
                n += 1
                file_ID = f"{pod_name2}_{ep_ID}_{n}"
                wav_dir = clip_path / file_ID




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
