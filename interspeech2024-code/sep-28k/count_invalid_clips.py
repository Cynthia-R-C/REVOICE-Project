# count_invalid_clips.py
# Cynthia Chen 12/6/2025
# Assumes the same folder structure and file naming as convert_sep28k_to_wenet.py

import csv
from pathlib import Path
import torchaudio

# Base path is the directory of this script (the sep-28k folder)
base = Path(__file__).resolve().parent

# Paths to the label CSV files (same as in convert_sep28k_to_wenet.py)
fl_label_csv = Path(r'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\raw_data\\SEP-28k\\fluencybank_labels.csv')
sep_label_csv = Path(r'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\raw_data\\SEP-28k\\SEP-28k_labels.csv')

# Path to audio clips (same as in convert_sep28k_to_wenet.py)
clip_path = Path(r'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\raw_data\\SEP-28k\\clips\\stuttering-clips\\clips')


def is_audio_valid(wav_path: Path) -> bool:
    '''Check if the audio file at wav_path is valid (non-empty and loadable).
    Returns True if valid, False otherwise.'''
    try:
        waveform, sr = torchaudio.load(wav_path)
        return waveform.numel() > 0
    except Exception:
        return False


def count_files_for_dataset(label_csv: Path, dataset_name: str):
    '''Count total, valid, and invalid audio files referenced in label_csv.'''
    total = 0
    invalid = 0

    with open(label_csv, newline='', encoding='utf-8') as flabel:
        label_reader = csv.DictReader(flabel)

        for row in label_reader:
            total += 1

            # Get Show, EpId, ClipId from CSV
            show = row.get('Show').strip()
            ep_ID = row.get('EpId').strip()
            clip_ID = row.get('ClipId').strip()

            # Example: FluencyBank_010_0  -> FluencyBank_010_0.wav
            file_ID = f'{show}_{ep_ID}_{clip_ID}'
            wav_path = clip_path / f'{file_ID}.wav'

            if not is_audio_valid(wav_path):
                invalid += 1

    valid = total - invalid
    invalid_pct = (invalid / total * 100) if total > 0 else 0.0

    print(f'{dataset_name}:')
    print(f'  Total clips in CSV: {total}')
    print(f'  Valid audio files:  {valid}')
    print(f'  Invalid audio files:{invalid} ({invalid_pct:.2f}% invalid)\n')

    return total, valid, invalid


if __name__ == '__main__':
    # Per-dataset counts
    fb_total, fb_valid, fb_invalid = count_files_for_dataset(fl_label_csv, 'FluencyBank')
    sep_total, sep_valid, sep_invalid = count_files_for_dataset(sep_label_csv, 'SEP-28k')

    # Overall counts across both datasets
    overall_total = fb_total + sep_total
    overall_valid = fb_valid + sep_valid
    overall_invalid = fb_invalid + sep_invalid
    overall_invalid_pct = (overall_invalid / overall_total * 100) if overall_total > 0 else 0.0

    print('Overall across FluencyBank + SEP-28k:')
    print(f'  Total clips referenced in CSVs: {overall_total}')
    print(f'  Valid audio files:              {overall_valid}')
    print(f'  Invalid audio files:            {overall_invalid} ({overall_invalid_pct:.2f}% invalid)\n')

    # Also report how many wav files are actually present on disk
    all_wavs = list(clip_path.glob('*.wav'))
    print(f'Actual wav files found on disk in {clip_path}: {len(all_wavs)}')
