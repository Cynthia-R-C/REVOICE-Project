"""
Tests ALL FIVE stutter types (/p, /b, /r, /wr, /i) against real, human,
SEP-28k-labeled clips - not just /p and /b. Fills in the two remaining unconfirmed
cases:
  1. ConvLSTM on all 5 types (previously only tested on a different synthesized file)
  2. StutterNet on /r, /wr, /i specifically (previously only /p, /b were tested)

Same robustness handling as test_real_labeled_clips.py: majority-vote (>=2 of 3
annotators) selection, file-existence + size pre-check, try/except around actual
loading with automatic replacement from the candidate pool, exact extract_clips.py
naming convention.

Usage:
    # Test ConvLSTM on all 5 types (edit CONFIG_PATH/CKPT_PATH/CMVN_PATH below first)
    python test_all_types.py --labels SEP-28k_labels.csv --clips clips

    # Then edit the path constants to StutterNet's and rerun to fill in /r, /wr, /i for it
"""

import argparse
import pathlib
import random

import librosa
import numpy as np
import pandas as pd

from destutterer import Destutterer

# ConvLSTM
CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_convlstm.yaml'
CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\convlstm_en\\63.pt'
CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\global_cmvn'

# StutterNet
# CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_stutternet.yaml'
# CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\stutternet_en\\36.pt'
# CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\stutternet_global_cmvn'

SR = 16000
WINDOW_SEC = 3.0
WINDOW_SAMPLES = int(WINDOW_SEC * SR)
MAJORITY_THRESHOLD = 2  # at least 2/3 annotators

# Dict for CSV column names
LABEL_COLUMNS = {
    "/p": "Prolongation",
    "/b": "Block",
    "/r": "SoundRep",
    "/wr": "WordRep",
    "/i": "Interjection",
}

def find_column(df, target_name):
    for col in df.columns:
        if col.strip().lower() == target_name.lower():
            return col
    return None

def clip_path_for(clips_dir, show, ep, clip_id):
    return clips_dir / show / ep / f"{show}_{ep}_{clip_id}.wav"

def clip_looks_present(path):
    '''File exists + nonzero size'''
    try:
        return path.exists() and path.stat().st_size > 100
    except OSError:
        return False

def load_clip(path):
    '''Load clip and force it to exactly WINDOW_SAMPLES (3s)'''
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    else:
        audio = audio[:WINDOW_SAMPLES]
    return audio.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Test the model on all 5 stutter types using actual SEP-28k clips.")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--clips", type=str, required=True)
    parser.add_argument("--n", type=int, default=8, help="Number of clips to test per category (default 8)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    clips_dir = pathlib.Path(args.clips)

    frames = [pd.read_csv(p, dtype={"EpId": str}) for p in args.labels]
    data = pd.concat(frames, ignore_index=True)

    show_col = find_column(data, "Show")
    ep_col = find_column(data, "EpId")
    clip_col = find_column(data, "ClipId")
    col_for = {code: find_column(data, name) for code, name in LABEL_COLUMNS.items()}
    stutter_cols = [c for c in col_for.values() if c]

    if not all([show_col, ep_col, clip_col]) or not all(col_for.values()):
        print("ERROR: couldn't find required columns. Found columns:")
        print(list(data.columns))
        return

    print(f"Loaded {len(data)} labeled clips from {len(args.labels)} file(s).")
    print(f"Testing model at: {CKPT_PATH}\n")

    def exists(row):
        return clip_looks_present(clip_path_for(clips_dir, str(row[show_col]), str(row[ep_col]).strip(), row[clip_col]))

    data["_exists"] = data.apply(exists, axis=1)
    available = data[data["_exists"]]
    print(f"{len(available)} of {len(data)} labeled clips have audio present on disk.\n")

    def candidate_pool(mask):
        subset = available[mask]
        if len(subset) == 0:
            return []
        return subset.sample(frac=1, random_state=args.seed).to_dict("records")

    print("Loading model...")
    d = Destutterer(CONFIG_PATH, CKPT_PATH, CMVN_PATH, sr=SR)
    print()

    def run_category(pool, label, target_key):
        '''target_key: which of the 5 prob outputs to track for this category
        None for the fluent baseline, which tracks all 5'''
        vals = {code: [] for code in LABEL_COLUMNS} if target_key is None else {target_key: []}
        n_skipped = 0
        for row in pool:
            if len(next(iter(vals.values()))) >= args.n:
                break
            path = clip_path_for(clips_dir, str(row[show_col]), str(row[ep_col]).strip(), row[clip_col])
            try:
                audio = load_clip(path)
            except Exception:
                n_skipped += 1
                continue
            probs = d.get_audio_stutter_probs(audio)
            for k in vals:
                vals[k].append(probs[k])
        n_got = len(next(iter(vals.values())))
        print(f"  {label}: {n_got}/{args.n} clips tested ({n_skipped} skipped as unloadable, pool had {len(pool)})")
        return vals

    print("=== Fluent baseline (all 5 probability types) ===")
    fluent_mask = (available[stutter_cols] == 0).all(axis=1)
    fluent_vals = run_category(candidate_pool(fluent_mask), "fluent", target_key=None)
    print()

    print("=== Per-type positive clips ===")
    results = {}
    for code, col in col_for.items():
        pool = candidate_pool(available[col] >= MAJORITY_THRESHOLD)
        vals = run_category(pool, f"{code} ({LABEL_COLUMNS[code]})-labeled", target_key=code)
        results[code] = vals[code]
    print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Type':<6} {'Positive-clip mean':>20} {'Fluent-clip mean':>20} {'Diff':>10} {'Verdict':>14}")
    print("-" * 78)
    for code in LABEL_COLUMNS:
        pos = results[code]
        flu = fluent_vals[code]
        if not pos or not flu:
            print(f"{code:<6} {'(no data)':>20}")
            continue
        pos_mean, flu_mean = np.mean(pos), np.mean(flu)
        diff = pos_mean - flu_mean
        # crude verdict threshold - a real difference should clearly exceed the kind
        # of noise we saw in the flat /p, /b results (~0.001-0.01)
        verdict = "DISCRIMINATES" if diff > 0.03 else "FLAT"
        print(f"{code:<6} {pos_mean:>20.4f} {flu_mean:>20.4f} {diff:>10.4f} {verdict:>14}")

if __name__ == "__main__":
    main()