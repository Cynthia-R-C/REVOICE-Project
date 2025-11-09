# V2
# Cynthia Chen 10/18/2025

#!/usr/bin/env python3
# convert_sep28k_to_wenet.py
# Usage: python convert_sep28k_to_wenet.py; assumes path of relevant CSV files; not applicable to other datasets or paths

import csv
from pathlib import Path
import torchaudio  # for checking audio validity

# ann_csv = sys.argv[1]
# wav_dir = Path(sys.argv[2])
# out = Path(sys.argv[3])
# out.mkdir(parents=True, exist_ok=True)

# Base path is the directory of this script
base = Path(__file__).resolve().parent  # sep-28k folder

# Paths to the csv files
# f_ep_csv = Path('../../raw_data/SEP-28k/fluencybank_episodes.csv')   # also these episodepaths are wrong, fix them with the base path if I wanna use them again
# sep_ep_csv = Path('../../raw_data/SEP-28k/SEP-28k_episodes.csv')
fl_label_csv = base.parent.parent / 'raw_data' / 'SEP-28k' / 'fluencybank_labels.csv'
sep_label_csv = base.parent.parent / 'raw_data' / 'SEP-28k' / 'SEP-28k_labels.csv'

# Path to audio clips
clip_path = base.parent.parent / 'raw_data' / 'SEP-28k' / 'clips' / 'stuttering-clips' / 'clips'

# Creating the output paths
out = base.parent / 'data' / 'train'  # base.parent is the interspeech2024-code folder
out.mkdir(parents=True, exist_ok=True)  # create the directory if not exist

wav_scp = out / "wav.scp"
segments = out / "segments"
text = out / "text"

# Helper function to detect empty audio files and not include them in the Kaldi files
def is_audio_valid(wav_path):
    '''Check if the audio file at wav_path is valid (non-empty).'''
    try:
        waveform, sr = torchaudio.load(wav_path)
        return waveform.numel() > 0
    except Exception as e:
        print(f"❌ Error loading audio file: {wav_path}\n{e}")
        return False


# Writes all the correct info for the dataset into the relevant files;
# Run twice, once for FluencyBank CSVs and one for SEP-28k CSVs

def convert_dataset(label_csv):
    '''Convert a given dataset with its episode and label CSV files into WeNet format for StutterNet training.'''

    # Read episode CSV to get wav directory
    with open(label_csv, newline='', encoding='utf-8') as flabel, \
        open(wav_scp, 'w', encoding='utf-8') as fwav, \
        open(segments, 'w', encoding='utf-8') as fseg, \
        open(text, 'w', encoding='utf-8') as ftxt:

        label_reader = csv.DictReader(flabel)

        # For each audio clip/row in the label CSV
        for row in label_reader:

            # Get show name, EpId, ClipId
            show = row.get('Show').strip()
            ep_ID = row.get('EpId').strip()
            clip_ID = row.get('ClipId').strip()

            # e.g. FluencyBank_010_0
            file_ID = f"{show}_{ep_ID}_{clip_ID}"

            # Check if audio is valid
            wav_dir = Path(f"{clip_path}/{file_ID}.wav")  # Formula for audio file path: Show_EpId_ClipId.wav
            if not is_audio_valid(wav_dir):
                print(f"⚠️ Skipping invalid audio file: {wav_dir}")
                continue  # skip this file if audio is invalid

            # Write fwav: rec_file_ID wav_dir
            fwav.write(f"{file_ID} {wav_dir}\n")

            # Write segments: utt_file_ID rec_file_ID 0 stop/16000-start/16000
            start = int(row.get('Start'))
            stop = int(row.get('Stop'))
            duration = (stop - start) / 16000.0  # assuming 16kHz sampling rate
            fseg.write(f"{file_ID} {file_ID} 0.00 {duration:.2f}\n")

            # Write text: utt_file_ID is_prolong,is_block,is_soundrep,is_wordrep,is_interj
            prolong_score = int(row.get('Prolongation'))
            is_prolong = 1 if prolong_score > 1 else 0
            block_score = int(row.get('Block'))
            is_block = 1 if block_score > 1 else 0
            soundrep_score = int(row.get('SoundRep'))
            is_soundrep = 1 if soundrep_score > 1 else 0
            wordrep_score = int(row.get('WordRep'))
            is_wordrep = 1 if wordrep_score > 1 else 0
            interj_score = int(row.get('Interjection'))
            is_interj = 1 if interj_score > 1 else 0

            #is_dys = 1 if (is_prolong or is_block or is_soundrep or is_wordrep or is_interj) else 0  # got rid of this to match StutterNet architecture

            ftxt.write(f"{file_ID} {is_prolong},{is_block},{is_soundrep},{is_wordrep},{is_interj}\n")

# Convert FluencyBank dataset
convert_dataset(fl_label_csv)
print("FluencyBank conversion completed.")

# Convert SEP-28k dataset
convert_dataset(sep_label_csv)
print("SEP-28k conversion completed.")