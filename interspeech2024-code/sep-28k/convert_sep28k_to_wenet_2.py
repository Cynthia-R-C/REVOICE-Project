import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd

# --- helper functions ---
def find_episode_basename(row):
    """
    Try to find a URL or filename in the given row (a pandas Series).
    We prefer columns that look like URLs (start with http) or end with common media extensions.
    """
    for val in row:
        if not isinstance(val, str):
            continue
        v = val.strip()
        if v.startswith("http://") or v.startswith("https://"):
            return Path(v).name  # basename from URL
        if v.lower().endswith((".wav", ".mp3", ".mp4", ".flac", ".m4a")):
            return Path(v).name
    return None

def detect_labels_columns(df_labels):
    """
    Identify the key columns in the labels dataframe.
    Returns a dict with keys: EpId, ClipId, Start, Stop, events (list of event column names).
    """
    cols = [c.strip() for c in df_labels.columns.astype(str).tolist()]
    lname = lambda choices: next((c for c in cols for t in choices if t.lower() in c.lower()), None)

    ep_col = lname(['epid', 'epid', 'ep', 'episode', 'show']) or 'EpId'
    clip_col = lname(['clipid', 'clip', 'utt', 'ClipId']) or 'ClipId'
    start_col = lname(['start', 'begin']) or 'Start'
    stop_col = lname(['stop', 'end']) or 'Stop'

    # event columns we care about (flexible matching)
    candidate_events = ['block', 'prolongation', 'soundrep', 'sound', 'wordrep', 'word', 'interjection', 'interj']
    events = []
    for c in cols:
        cl = c.lower()
        for ev in candidate_events:
            if ev in cl and c not in events:
                events.append(c)
    # ensure common order: Block, Prolongation, SoundRep, WordRep, Interjection
    preferred = ['block', 'prolongation', 'soundrep', 'wordrep', 'interjection']
    ordered_events = []
    for p in preferred:
        found = next((c for c in events if p in c.lower()), None)
        if found:
            ordered_events.append(found)
    # if some missing, append any remaining candidate event columns
    for c in events:
        if c not in ordered_events:
            ordered_events.append(c)
    return {'EpId': ep_col, 'ClipId': clip_col, 'Start': start_col, 'Stop': stop_col, 'events': ordered_events}

def convert_start_stop_to_seconds(start, stop, sample_rate=16000):
    """
    Heuristically convert Start/Stop values to seconds.
    If values look large (>1e6), assume they are sample indices and divide by sample_rate.
    If they look like milliseconds (>=1000 and <= 1e6), divide by 1000.
    Otherwise assume they are seconds already.
    """
    if start is None or stop is None:
        return None, None
    try:
        s = float(start)
        e = float(stop)
    except Exception:
        return None, None
    mx = max(abs(s), abs(e))
    if mx > 1e6:
        # likely sample indices
        return s / sample_rate, e / sample_rate
    if 1000 <= mx <= 1e6:
        # likely milliseconds
        return s / 1000.0, e / 1000.0
    # otherwise assume seconds already
    return s, e

# --- main conversion logic ---
def convert(episodes_csv, labels_csv, wav_dir, out_dir, episode_exts=(".wav", ".mp4", ".mp3"), sample_rate=16000, make_shards=False):
    episodes_csv = Path(episodes_csv)
    labels_csv = Path(labels_csv)
    wav_dir = Path(wav_dir) if wav_dir else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSVs robustly
    # Episodes file: it may not have headers; read with header=None and try to detect useful columns.
    try:
        ep_df = pd.read_csv(episodes_csv, skipinitialspace=True, header=0)
    except Exception:
        ep_df = pd.read_csv(episodes_csv, header=None)
    # Labels should have headers, but be robust
    lab_df = pd.read_csv(labels_csv, skipinitialspace=True, header=0)

    # Build mapping EpId -> episode basename (e.g. 46ma.mp4)
    ep_map = {}
    # If episodes file has sensible column names, try to detect keys
    if 'EpId' in ep_df.columns or 'EpId' in [c.strip() for c in ep_df.columns]:
        # normalize column names
        cols = {c: c.strip() for c in ep_df.columns}
        ep_df = ep_df.rename(columns=cols)
    # iterate rows, attempt to find EpId and URL/filename
    for _, row in ep_df.iterrows():
        # Convert row to Series of strings for detection
        name = find_episode_basename(row)
        # try to find an EpId value in the row (numeric-like)
        epid = None
        for col in row.index:
            v = row[col]
            if pd.isna(v):
                continue
            sval = str(v).strip()
            if sval.isdigit():
                epid = sval.lstrip('0') or '0'  # strip leading zeros
                break
        # fallback: if episodes file had a column named EpId, use it
        if epid is None and 'EpId' in ep_df.columns:
            epid = str(row['EpId']).strip()
        if epid is None:
            # skip if we couldn't detect EpId
            continue
        if name is None:
            # if no URL/filename detected, optionally continue; here we skip
            continue
        # store mapping: epid -> basename name (without path)
        ep_map[epid] = Path(name).name

    if len(ep_map) == 0:
        print("Warning: no episodes detected from episodes CSV. EpId->filename map is empty.", file=sys.stderr)

    # Detect label columns
    cols = detect_labels_columns(lab_df)
    ep_col = cols['EpId']
    clip_col = cols['ClipId']
    start_col = cols['Start']
    stop_col = cols['Stop']
    event_cols = cols['events']

    if len(event_cols) == 0:
        print("Warning: no event columns auto-detected in labels CSV. Expected columns like Block, Prolongation, SoundRep, WordRep, Interjection.", file=sys.stderr)

    # We'll create these files
    wav_scp_path = out_dir / "wav.scp"
    segments_path = out_dir / "segments"
    text_path = out_dir / "text"
    data_list_path = out_dir / "data.list"

    # We'll map episode recording id to absolute path to file in wav_dir
    recid_to_path = {}

    # Helper to find local episode file by basename in wav_dir
    def find_local_episode(basename):
        if not wav_dir:
            return None
        # try with known extensions
        base = Path(basename)
        name_noext = base.stem
        for ext in episode_exts:
            cand = wav_dir / (name_noext + ext)
            if cand.exists():
                return cand.resolve()
        # try exact basename
        cand = wav_dir / basename
        if cand.exists():
            return cand.resolve()
        return None

    # Prepare outputs
    wav_lines = []
    seg_lines = []
    text_lines = []
    data_list = []

    # Iterate labels and produce utt entries
    for idx, row in lab_df.iterrows():
        # get ep id and clip id
        epid = str(row.get(ep_col) if ep_col in row else row[ep_col]) if ep_col in row or ep_col in lab_df.columns else None
        clipid = str(row.get(clip_col) if clip_col in row else row[clip_col]) if clip_col in row or clip_col in lab_df.columns else None
        if epid is None or clipid is None:
            continue
        epid = epid.strip()
        clipid = clipid.strip()
        # find episode basename from ep_map or try to derive from ep id
        basename = ep_map.get(epid, None)
        if basename is None:
            # try to use EpId as file name (common in SEP-28k where filenames are like 46ma.mp4)
            # if ep_df had a URL column, ep_map would have the basename; otherwise we leave as None
            pass

        # Build recording id and utt id
        rec_id = f"ep{epid}"
        utt_id = f"{rec_id}_c{clipid}"

        # Build wav path for this recording
        if rec_id not in recid_to_path:
            abs_path = None
            if basename:
                abs_path = find_local_episode(basename)
            if abs_path is None and wav_dir is not None:
                # as fallback, try to find any file in wav_dir that contains the epid (e.g. '46ma', '010')
                candidates = list(wav_dir.glob(f"*{epid}*"))
                if candidates:
                    abs_path = candidates[0].resolve()
            if abs_path is None:
                # leave as a placeholder; user can replace or download episodes into wav_dir
                abs_path = Path(f"<PATH_TO_EPISODE_FOR_{rec_id}>").resolve()
            recid_to_path[rec_id] = abs_path

        # compute start and stop seconds heuristically
        start_raw = row.get(start_col) if start_col in row else row[start_col]
        stop_raw = row.get(stop_col) if stop_col in row else row[stop_col]
        start_sec, stop_sec = convert_start_stop_to_seconds(start_raw, stop_raw, sample_rate=sample_rate)
        if start_sec is None or stop_sec is None:
            # skip if cannot determine segment times
            continue
        # make sure end > start
        if stop_sec <= start_sec:
            continue

        # collect event values
        ev_vals = []
        for evcol in event_cols:
            v = row.get(evcol) if evcol in row else row[evcol] if evcol in lab_df.columns else 0
            try:
                iv = int(float(v))
            except Exception:
                # try parse from strings like '0' or ' 0'
                try:
                    iv = int(str(v).strip() or 0)
                except Exception:
                    iv = 0
            # convert counts >1 to 1 (we care about presence)
            ev_vals.append(1 if iv > 0 else 0)

        # Ensure we have five event columns; pad with zeros if necessary
        while len(ev_vals) < 5:
            ev_vals.append(0)
        # Order: Block, Prolongation, SoundRep, WordRep, Interjection
        # If auto-detection found different order, we assume it's already in sensible order based on detect_labels_columns.
        # Compute is_dysfluent
        is_dys = 1 if any(ev_vals[:5]) else 0

        # Write wav.scp entry (once per rec_id)
        wav_lines.append((rec_id, str(recid_to_path[rec_id])))

        # segments: utt_id rec_id start end (rounded to 2 decimals)
        seg_lines.append((utt_id, rec_id, f"{start_sec:.2f}", f"{stop_sec:.2f}"))

        # text: utt_id labelstring (is_dys then 5 events)
        label_str = f"{is_dys} " + " ".join(str(int(x)) for x in ev_vals[:5])
        text_lines.append((utt_id, label_str))

        # convenience data.list JSON line
        data_list.append({"key": utt_id, "wav": str(recid_to_path[rec_id]), "start": round(start_sec, 2), "end": round(stop_sec, 2), "txt": label_str})

    # Deduplicate wav_lines (keep first occurrence)
    seen = set()
    wav_lines_unique = []
    for rec, p in wav_lines:
        if rec not in seen:
            wav_lines_unique.append((rec, p))
            seen.add(rec)

    # Write files
    with open(wav_scp_path, "w", encoding="utf-8") as f:
        for rec, p in wav_lines_unique:
            f.write(f"{rec} {p}\n")
    with open(segments_path, "w", encoding="utf-8") as f:
        for utt, rec, s, e in seg_lines:
            f.write(f"{utt} {rec} {s} {e}\n")
    with open(text_path, "w", encoding="utf-8") as f:
        for utt, lab in text_lines:
            f.write(f"{utt} {lab}\n")
    with open(data_list_path, "w", encoding="utf-8") as f:
        for obj in data_list:
            f.write(json.dumps(obj) + "\\n")

    print(f"Wrote {wav_scp_path}, {segments_path}, {text_path}, and {data_list_path}.")
    if make_shards:
        # Try to call tools/make_shard_list.py (best-effort)
        import subprocess, shutil
        script = shutil.which("python") or shutil.which("python3") or "python"
        make_shard = Path("tools") / "make_shard_list.py"
        if make_shard.exists():
            cmd = [script, str(make_shard), "--num_utts_per_shard", "500", "--num_threads", "8", "--segments", str(segments_path), str(wav_scp_path), str(text_path), str(out_dir / "shards"), str(out_dir / "data.list")]
            print("Running:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                print("make_shard_list.py completed.")
            except subprocess.CalledProcessError as e:
                print("make_shard_list.py failed:", e, file=sys.stderr)
        else:
            print("tools/make_shard_list.py not found at ./tools. Skipping shard creation.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert FluencyBank/SEP-28k CSVs to Kaldi-style data files")
    p.add_argument("--episodes-csv", required=True, help="Path to episodes CSV (episode-level metadata with URL or filename)")
    p.add_argument("--labels-csv", required=True, help="Path to labels CSV (per-clip labels)")
    p.add_argument("--wav-dir", required=False, default=None, help="Local directory containing episode audio files (basenames should match URLs from episodes CSV). If not provided, wav.scp will contain placeholders.")
    p.add_argument("--out-dir", required=True, help="Output data directory to write wav.scp, segments, text, data.list")
    p.add_argument("--sample-rate", type=int, default=16000, help="Sample rate to use when interpreting sample-index Start/Stop values (default 16000).")
    p.add_argument("--make-shards", action="store_true", help="If set and tools/make_shard_list.py exists, run it to produce shards and a final data.list")
    args = p.parse_args()
    convert(args.episodes_csv, args.labels_csv, args.wav_dir, args.out_dir, sample_rate=args.sample_rate, make_shards=args.make_shards)