# Cynthia Chen 7/9/2026
# Counts how many valid audio clips per stutter type exist in the ml-stuttering-events-dataset (new) SEP-28k+FluencyBank dataset

import argparse
import pathlib
import torchaudio  # for checking audio validity

import pandas as pd

LABEL_COLUMNS = {
    "/p": "Prolongation",
    "/b": "Block",
    "/r": "SoundRep",
    "/wr": "WordRep",
    "/i": "Interjection",
}

MAJORITY_THRESHOLD = 2  # at least 2/3 annotators agree


def find_column(df, target_name):
    """Case-insensitive column lookup, since CSV column casing can vary by source."""
    for col in df.columns:
        if col.strip().lower() == target_name.lower():
            return col
    return None


def main():
    parser = argparse.ArgumentParser(description="Count stutter-type labels for clips that actually exist on disk.")
    parser.add_argument("--labels", nargs="+", required=True,
                         help="One or more label CSV files (e.g. SEP-28k_labels.csv fluencybank_labels.csv)")
    parser.add_argument("--clips", type=str, required=True,
                         help="Path to the extracted clips directory (the --clips argument you gave extract_clips.py)")
    args = parser.parse_args()

    clips_dir = pathlib.Path(args.clips)

    frames = []
    for path in args.labels:
        # dtype={"EpId": str} matches extract_clips.py
        df = pd.read_csv(path, dtype={"EpId": str})
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    show_col = find_column(data, "Show")
    ep_col = find_column(data, "EpId")
    clip_col = find_column(data, "ClipId")
    if not all([show_col, ep_col, clip_col]):
        print("ERROR: couldn't find Show/EpId/ClipId columns. Found columns:")
        print(list(data.columns))
        return

    print(f"Loaded {len(data)} labeled clips from {len(args.labels)} file(s).")

    # Helper function to detect empty audio files and not include them in the Kaldi files
    def is_audio_valid(wav_path):
        '''Check if the audio file at wav_path is valid (non-empty).'''
        try:
            waveform, sr = torchaudio.load(wav_path)
            return waveform.numel() > 0
        except Exception as e:
            print(f"❌ Error loading audio file: {wav_path}\n{e}")
            return False

    # Check which clips actually exist on disk
    # extract_clips.py's naming conv: {clips_dir}/{Show}/{EpId}/{Show}_{EpId}_{ClipId}.wav
    def clip_exists(row):
        show = str(row[show_col])
        ep = str(row[ep_col]).strip()
        clip_id = row[clip_col]
        clip_path = clips_dir / show / ep / f"{show}_{ep}_{clip_id}.wav"
        if clip_path.exists():
            if not is_audio_valid(clip_path):
                print(f"⚠️ Skipping invalid audio file: {clip_path}")
                return False
        return clip_path.exists()

    exists_mask = data.apply(clip_exists, axis=1)
    n_total = len(data)
    data_with_audio = data[exists_mask].copy()
    n_have = len(data_with_audio)

    print(f"Clips that actually exist in {args.clips}: {n_have} of {n_total} ({n_have / n_total:.1%})\n")

    print(f"{'Type':<6} {'Column':<14} {'All labels':>12} {'Have audio':>12} {'Retained %':>12}")
    print("-" * 60)

    retained_pcts = {}
    for short_code, col_name in LABEL_COLUMNS.items():
        actual_col = find_column(data, col_name)
        if actual_col is None:
            print(f"{short_code:<6} {col_name:<14} {'COLUMN NOT FOUND':>12}")
            continue
        votes_all = pd.to_numeric(data[actual_col], errors="coerce").fillna(0)
        votes_have = pd.to_numeric(data_with_audio[actual_col], errors="coerce").fillna(0)
        count_all = (votes_all >= MAJORITY_THRESHOLD).sum()
        count_have = (votes_have >= MAJORITY_THRESHOLD).sum()
        retained_pct = count_have / count_all if count_all else 0
        retained_pcts[short_code] = retained_pct
        print(f"{short_code:<6} {col_name:<14} {count_all:>12} {count_have:>12} {retained_pct:>11.1%}")

    print()
    if retained_pcts:
        avg_retained = sum(retained_pcts.values()) / len(retained_pcts)
        print(f"Average retention across all 5 types: {avg_retained:.1%}")
        print()
        for short_code, pct in sorted(retained_pcts.items(), key=lambda x: x[1]):
            diff = pct - avg_retained
            if abs(diff) > 0.03:  # flag anything more >3 pt off the average
                direction = "BELOW" if diff < 0 else "ABOVE"
                print(f"  {short_code} retained {pct:.1%} - {direction} average by {abs(diff):.1%}")


if __name__ == "__main__":
    main()

