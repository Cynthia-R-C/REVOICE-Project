"""
Runs the model against REAL, human, SEP-28k-labeled clips - some genuinely labeled
/p (Prolongation), some genuinely labeled /b (Block), and some genuinely fluent -
to test whether the flat, non-discriminating output we found earlier is specific to
the ElevenLabs TTS-simulated stutter file, or a real problem with the trained model
itself on the exact kind of data it was trained on.

Uses the standard SEP-28k majority-vote convention (>=2 of 3 annotators) to select
genuinely positive/fluent examples, and only picks clips that actually exist on disk
(same file-existence check as count_stutter_types.py) - matching the exact naming
convention from Apple's own extract_clips.py:
    {clips_dir}/{Show}/{EpId}/{Show}_{EpId}_{ClipId}.wav

Usage:
    python test_real_labeled_clips.py --labels SEP-28k_labels.csv --clips clips
    python test_real_labeled_clips.py --labels SEP-28k_labels.csv --clips clips --n 8
"""

import argparse
import pathlib
import random

import librosa
import numpy as np
import pandas as pd

from destutterer import Destutterer

# --- adjust to whichever model you want to test (StutterNet shown, matches the
# earlier real-clip test) - swap to the ConvLSTM paths to test that model instead ---
CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_stutternet.yaml'
CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\stutternet_en\\36.pt'
CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\stutternet_global_cmvn'
SR = 16000
WINDOW_SEC = 3.0
WINDOW_SAMPLES = int(WINDOW_SEC * SR)

MAJORITY_THRESHOLD = 2  # at least 2 of 3 annotators agree - standard SEP-28k convention


def find_column(df, target_name):
    for col in df.columns:
        if col.strip().lower() == target_name.lower():
            return col
    return None


def clip_path_for(clips_dir, show, ep, clip_id):
    return clips_dir / show / ep / f"{show}_{ep}_{clip_id}.wav"


def clip_looks_present(path):
    """Cheap upfront filter: file exists AND has nonzero size. Catches the common
    case of a clip whose Start/Stop offsets fell past the end of a truncated/short
    episode download, which extract_clips.py can leave behind as an empty or
    near-empty file rather than erroring out. Not a full validity guarantee on its
    own - actually loading the file (see load_clip, called with try/except below)
    is still the real check, but this avoids wasting time even attempting to load
    obviously-empty files."""
    try:
        return path.exists() and path.stat().st_size > 100
    except OSError:
        return False


def load_clip(path):
    """Load a clip and force it to exactly WINDOW_SAMPLES (3.0s) - SEP-28k clips
    should already be ~3s, but pad/truncate defensively since encoding can shift
    exact sample counts by a few frames. Raises on genuinely invalid/corrupt audio -
    the caller is responsible for catching that and skipping the clip."""
    # str(path) rather than passing the Path object directly - some soundfile
    # versions only accept a plain string and reject PathLike objects with a
    # generic "Invalid file" TypeError, which is indistinguishable from an
    # actually-corrupt file unless you know to check for this specifically.
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    else:
        audio = audio[:WINDOW_SAMPLES]
    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Test the model on real, human, SEP-28k-labeled clips.")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--clips", type=str, required=True)
    parser.add_argument("--n", type=int, default=6, help="Number of clips to test per category (default 6)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    clips_dir = pathlib.Path(args.clips)

    frames = [pd.read_csv(p, dtype={"EpId": str}) for p in args.labels]
    data = pd.concat(frames, ignore_index=True)

    show_col = find_column(data, "Show")
    ep_col = find_column(data, "EpId")
    clip_col = find_column(data, "ClipId")
    p_col = find_column(data, "Prolongation")
    b_col = find_column(data, "Block")
    r_col = find_column(data, "SoundRep")
    wr_col = find_column(data, "WordRep")
    i_col = find_column(data, "Interjection")

    stutter_cols = [c for c in [p_col, b_col, r_col, wr_col, i_col] if c]

    def exists(row):
        return clip_looks_present(clip_path_for(clips_dir, str(row[show_col]), str(row[ep_col]).strip(), row[clip_col]))

    data["_exists"] = data.apply(exists, axis=1)
    available = data[data["_exists"]]
    print(f"{len(available)} of {len(data)} labeled clips have audio present on disk (nonzero-size file found).\n")
    print("Note: this still doesn't guarantee every file is fully valid, loadable audio -")
    print("some clips (e.g. from a truncated/short episode download) can exist as a")
    print("nonzero-size but corrupt file. Those get skipped individually below instead of")
    print("crashing the whole run, and are counted separately in the summary.\n")

    def candidate_pool(mask, label):
        subset = available[mask]
        if len(subset) == 0:
            print(f"WARNING: no available clips found for '{label}' - skipping.")
            return []
        # Shuffle the FULL pool of candidates (not just args.n) so run_category()
        # can draw replacements from it if some turn out to be unloadable.
        return subset.sample(frac=1, random_state=args.seed).to_dict("records")

    prolongation_pool = candidate_pool(available[p_col] >= MAJORITY_THRESHOLD, "prolongation")
    block_pool = candidate_pool(available[b_col] >= MAJORITY_THRESHOLD, "block")
    fluent_mask = (available[stutter_cols] == 0).all(axis=1)
    fluent_pool = candidate_pool(fluent_mask, "fluent")

    print("Loading model...")
    d = Destutterer(CONFIG_PATH, CKPT_PATH, CMVN_PATH, sr=SR)
    print()

    def run_category(pool, label):
        print(f"=== {label} clips (target: {args.n}) ===")
        p_vals, b_vals = [], []
        n_skipped = 0
        for row in pool:
            if len(p_vals) >= args.n:
                break
            path = clip_path_for(clips_dir, str(row[show_col]), str(row[ep_col]).strip(), row[clip_col])
            ep_display = str(row[ep_col]).strip()
            try:
                audio = load_clip(path)
            except Exception as e:
                n_skipped += 1
                print(f"  SKIPPED (unloadable/corrupt): {row[show_col]}/{ep_display}/clip{row[clip_col]}  ({e})")
                continue
            probs = d.get_audio_stutter_probs(audio)
            p_vals.append(probs['/p'])
            b_vals.append(probs['/b'])
            true_labels = f"Prolongation={row[p_col]} Block={row[b_col]} SoundRep={row[r_col]} WordRep={row[wr_col]} Interjection={row[i_col]}"
            print(f"  {row[show_col]}/{ep_display}/clip{row[clip_col]}  model: /p={probs['/p']:.4f} /b={probs['/b']:.4f}   true: {true_labels}")
        if len(p_vals) < args.n:
            print(f"  NOTE: only found {len(p_vals)} usable clips out of {args.n} requested "
                  f"(pool had {len(pool)} candidates, {n_skipped} were unloadable).")
        if p_vals:
            print(f"  --> mean /p = {np.mean(p_vals):.4f}   mean /b = {np.mean(b_vals):.4f}")
        print()
        return p_vals, b_vals

    prolongation_p, prolongation_b = run_category(prolongation_pool, "PROLONGATION-labeled")
    block_p, block_b = run_category(block_pool, "BLOCK-labeled")
    fluent_p, fluent_b = run_category(fluent_pool, "FLUENT")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if prolongation_p and fluent_p:
        print(f"/p on real prolongation clips: {np.mean(prolongation_p):.4f}   vs fluent: {np.mean(fluent_p):.4f}")
    if block_b and fluent_b:
        print(f"/b on real block clips:        {np.mean(block_b):.4f}   vs fluent: {np.mean(fluent_b):.4f}")
    print()
    print("If prolongation/block means are clearly higher than fluent means here (on")
    print("real human data), the model DOES work - the earlier flat result was specific")
    print("to the ElevenLabs TTS file, and retraining is not the priority; the deployment")
    print("side (windowing/thresholds/testing methodology) is.")
    print()
    print("If these numbers are just as flat/indistinguishable as they were on the TTS")
    print("file, that's a real, separate finding: the model doesn't discriminate even on")
    print("its own training-domain data, and retraining (with the retention-imbalance")
    print("fix for /p and /b) is genuinely the right next step.")


if __name__ == "__main__":
    main()