# Destutterer class to handle destuttering using StutterNet model
# Implemented in whisper_online_server.py
# Cynthia Chen 11/30/2025

import torch
import math
import numpy as np
import re

# Inference imports
import sys
import os
bin_dir = os.path.abspath('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\wenet\\wenet\\bin')
sys.path.append(bin_dir)  # add destutter folder to paths to search
from infer_sed_single import StutterSED

# Tuning
CROSSFADE = False  # whether or not to use crossfade
FADE = 0.015  # 15 ms fade for crossfade_concat
TARGET_KEEP = 0.12  # target keep length for /p destutter in seconds, tune by ear (e.g. 0.10–0.15)
PAUSE = 0.10  # target pause length for /b destutter in seconds, tune by ear (e.g. 0.08–0.12)

# Audio destutter guards to reduce false positives
# Req this many consecutive above-threshold windows before opening a region
# At hop=0.25s, MIN_CONSECUTIVE=2 means the stutter must persist for >=0.5s
MIN_CONSECUTIVE = 2
# Only apply a cut if detected region is at least this long (seconds)
# Real prolongations/blocks are sustained, brief spikes are noise
MIN_REGION_DUR = 0.5

# Override thresholds for /p and /b used in aud_destutter_chunk (pre-STT context)
# The thresholds.pt values were optimised for F-beta on the stutter eval set but produced too many false positives on fluent speech
# Run calibrate_aud_thresholds() on known fluent audio to find good values then set them here
# None = fall back to thresholds.pt values (not recommended).
AUD_THRESH_P = 0.368   # e.g. 0.75 - set after running calibration
AUD_THRESH_B = 0.479   # e.g. 0.80 - set after running calibration

# Debug prints flag
DEBUG = True


class Destutterer:
    '''Class to handle destuttering using StutterNet model'''

    def __init__(self, config_path, ckpt_path, cmvn_path, sr=16000, device='cpu'):
        '''Declare and initialize all needed parameters
        For StutterSED model: aud_type and override_config go by defaut parameters (arr and []) so no need to initialize here'''

        # Segment-specific instance variables are declared and intialized in get_destutter_info()
        self.t_to_buffer = None  # time between global stream start and audio buffer start
        self.audio_buffer = None  # current audio buffer - a numpy array of samples
        self.beg_time = None  # global start time of text (s)
        self.end_time = None  # global end time of text (s)
        self.text = ''  # text segment
        self.words = []  # list of words in text segment

        # Persistent rolling buffer for AUDIO destuttering only
        # Separate from self.audio_buffer above (which is a reference to Whisper's buffer)
        # Enough total buffered audio (retain_margin, below) that MULTIPLE overlapping 3.0s windows exist at once
        self._aud_buffer = np.array([], dtype=np.float32)

        # Non-segment-specific instance variables are initialized here
        self.config = config_path
        self.checkpoint = ckpt_path
        self.cmvn = cmvn_path
        self.sr = sr # sampling rate
        self.device = device

        # Load class StutterSED
        self.stutter_model = StutterSED(self.config, self.checkpoint, self.cmvn, gpu=0 if self.device=='cuda' else -1)

        # Load thresholds
        THRESH_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\eval\\thresholds.pt'
        self.thresholds_tensor = torch.load(THRESH_PATH).to(self.device)  # load computed thresholds

    def get_text(self):
        '''Returns current text'''
        return self.text

    def _run_aud_detection_and_cuts(self):
        '''Shared by aud_destutter_chunk() and flush_aud_buffer(): runs the sliding
        window over self._aud_buffer, finds /p and /b regions, and applies cuts
        in place. Assumes the caller has already appended any new audio.'''
        hop_samples = int(0.5 * self.sr)
        window_samples = int(3.0 * self.sr)

        if len(self._aud_buffer) < window_samples:
            return  # not enough audio for even one full window - nothing to detect

        window_probs = {'/p': [], '/b': []}
        center_times = []  # in seconds, relative to start of self._aud_buffer

        for win_start in range(0, len(self._aud_buffer) - window_samples + 1, hop_samples):
            win_end = win_start + window_samples
            segment = self._aud_buffer[win_start:win_end]
            if DEBUG:
                # TEMP DIAGNOSTIC: confirms whether each window's audio is genuinely different or if something upstream is feeding the model the same content every time
                # Doing this bc /p and /b probs are staying the same every slice
                print(f'[AUD_DESTUT DEBUG] window @ {win_start/self.sr:.2f}s  '
                      f'min={segment.min():.4f} max={segment.max():.4f} '
                      f'rms={np.sqrt(np.mean(segment.astype(np.float64)**2)):.4f} '
                      f'dtype={segment.dtype}')
            probs = self.get_audio_stutter_probs(segment)
            window_probs['/p'].append(probs['/p'])
            window_probs['/b'].append(probs['/b'])
            center_times.append((win_start + win_end) / 2.0 / self.sr)

        thresh_p = AUD_THRESH_P if AUD_THRESH_P is not None else self.thresholds_tensor[0].item()
        thresh_b = AUD_THRESH_B if AUD_THRESH_B is not None else self.thresholds_tensor[1].item()

        if DEBUG:
            print(f'[AUD_DESTUT] /p probs: {[f"{p:.2f}" for p in window_probs["/p"]]}  thresh={thresh_p:.2f}')
            print(f'[AUD_DESTUT] /b probs: {[f"{p:.2f}" for p in window_probs["/b"]]}  thresh={thresh_b:.2f}')

        p_start, p_end = self.estimate_region_local(center_times, window_probs['/p'], thresh_p)
        b_start, b_end = self.estimate_region_local(center_times, window_probs['/b'], thresh_b)

        # Temporarily set beg_time=0 so seg_local_to_global in debug prints works.
        self.beg_time = 0.0

        # Potential problem: if both a /p and a /b region are detected in the same pass, p_destutter is applied first and can shift sample indices before b_destutter runs against the (still pre-shift) b_start/b_end
        if p_start is not None and (p_end - p_start) >= MIN_REGION_DUR:
            self._aud_buffer = self.p_destutter(self._aud_buffer, p_start, p_end)
        elif p_start is not None and DEBUG:
            print(f'[AUD_DESTUT] /p region {p_start:.2f}s–{p_end:.2f}s skipped (duration {p_end - p_start:.2f}s < MIN_REGION_DUR {MIN_REGION_DUR}s)')

        if b_start is not None and (b_end - b_start) >= MIN_REGION_DUR:
            self._aud_buffer = self.b_destutter(self._aud_buffer, b_start, b_end)
        elif b_start is not None and DEBUG:
            print(f'[AUD_DESTUT] /b region {b_start:.2f}s–{b_end:.2f}s skipped (duration {b_end - b_start:.2f}s < MIN_REGION_DUR {MIN_REGION_DUR}s)')

    def aud_destutter_chunk(self, audio_chunk):
        '''Pre-STT audio destuttering for prolongations and blocks
        Maintains its own rolling buffer (self._aud_buffer) which guarentees multiple windows exist at once

        Appends to the internal buffer + runs detection/destuttering against the buffer.
        IMPORTANT: call flush_aud_buffer() once when stream ends to release remaining buffered tail

        Uses calibrated thresholds (AUD_THRESH_P / AUD_THRESH_B) if set, otherwise fall back to thresholds.pt values

        audio_chunk: 1D np.float32 array at self.sr
        Returns: 1D np.float32 array'''

        window_samples = int(3.0 * self.sr)
        hop_samples = int(0.5 * self.sr)
        
        retain_margin = window_samples + MIN_CONSECUTIVE * hop_samples  # retain enough margin

        # Append the new audio onto our persistent buffer
        if len(self._aud_buffer):
            self._aud_buffer = np.concatenate([self._aud_buffer, audio_chunk])
        else:
            self._aud_buffer = audio_chunk.astype(np.float32, copy=True)

        self._run_aud_detection_and_cuts()

        # Release everything except retain_margin at the tail 
        # Must stay buffered bc any of it could still be reevaluated by a future window once more audio arrives - a region only "closes" (and only gets cut) once
        release_len = len(self._aud_buffer) - retain_margin
        if release_len <= 0:
            return np.array([], dtype=np.float32)

        released = self._aud_buffer[:release_len]
        self._aud_buffer = self._aud_buffer[release_len:]
        return released

    def flush_aud_buffer(self):
        '''Call once at end-of-stream (client disconnected, no more audio coming) to release whatever's left in the rolling audio-destutter buffer
        
        Does NOT hold back a lookahead margin - there is no more audio coming, so there's nothing left to wait for'''
        if len(self._aud_buffer) == 0:
            return np.array([], dtype=np.float32)

        self._run_aud_detection_and_cuts()

        released = self._aud_buffer
        self._aud_buffer = np.array([], dtype=np.float32)
        return released

    def reset_aud_buffer(self):
        '''Call at the start of a new session to clear buffer'''
        self._aud_buffer = np.array([], dtype=np.float32)

    def get_destutter_info(self, client_type, t_to_buffer, audio_buffer, beg_time, end_time, text):
        '''Returns info needed for destuttering to call individual stutter_type methods separately
        Returns different info based on stt or tts destuttering
        stt: sound rep, word rep, interjections
        tts: prolongations, blocks
        
        t_to_buffer  # time between global stream start and audio buffer start
        audio_buffer  # current audio buffer - a numpy array of samples
        beg_time  # global start time of text (s)
        end_time  # global end time of text (s)
        text  # text segment'''

        # CODE NEEDED FOR ALL DESTUTTERING

        # Initialize all instance variables with their new values
        self.t_to_buffer = t_to_buffer  # time between global stream start and audio buffer start
        self.audio_buffer = audio_buffer  # current audio buffer - a numpy array of samples
        self.beg_time = beg_time  # global start time of text (s)
        self.end_time = end_time  # global end time of text (s)
        self.text = text  # text segment
        self.words = text.split() # list of words in text segment

        # Sliding 3s window over audio corresponding to text segment
        hop = 0.35 * self.sr   # could increase this if latency is too high
        window_size = 3.0 * self.sr  # 3s window; window size in samples
        loc_start = (self.beg_time - self.t_to_buffer) * self.sr  # buffer local
        loc_end = (self.end_time - self.t_to_buffer) * self.sr  # buffer local

        # Guard: if loc_end - loc_start < hop
        if loc_end - loc_start < hop:
            if client_type == 'stt':
                # No windows, no stutter detected
                t_maxs = {'/p': None, '/b': None, '/r': None, '/wr': None, '/i': None}
                stutter_word_idxs = {}
                return t_maxs, stutter_word_idxs
            else:  # tts
                return (None, None), (None, None)

        window_probs = {'/p': [], '/b': [], '/r': [], '/wr': [], '/i': []}
        center_ts = []  # local center times of each window (in seconds)

        # Getting the stutter probabilities for each window
        for win_start in range(int(loc_start), int(loc_end - window_size), int(hop)):
            audio_segment = self.audio_buffer[int(win_start):int(win_start + window_size)]  # get audio segment corresponding to text
            stutter_probs = self.get_audio_stutter_probs(audio_segment)  # get stutter probabilities for this audio segment
            # sttuter_probs is like {'/p': 0.7, '/b': 0.2, ...}
            
            # Store stutter probabilities for this window
            for label in window_probs.keys():
                window_probs[label].append(stutter_probs[label])  # append the prob for this window
                # window_probs is like {'/p': [p0, p1, ...], '/b': [...], ...}

            # Compute center time of the current window
            center_t = self.get_center_times_local(win_start, win_start + window_size)
            center_ts.append(center_t)
        
        # Find the times of max probs for each label

        # STT: care about /r, /wr, /i
        if client_type == 'stt':   # note to self: can later modify this to only calculate values for needed stt stutter types to reduce latency

            # Thresholds for relevant stutter types
            thresholds = {'/r': self.thresholds_tensor[2].item(),
                          '/wr': self.thresholds_tensor[3].item(),
                          '/i': self.thresholds_tensor[4].item() }  # convert to dict for easier access

            # Get times most likely to be a type of stutter
            t_maxs = self.get_max_times_local(center_ts, window_probs, thresholds)  # looks like: {'/p': 12.3, '/b': None, ...}
            # So by now via t_maxs it is has been determined whether a stutter type is actually present or not

            # Approximate local start/end times for each word in the text segment
            segment_duration = self.end_time - self.beg_time
            word_times_local = self.estimate_word_times_local(segment_duration)

            # if any stutter detected, find the corresponding word index(s)
            stutter_word_idxs = self.get_stutter_times_local(word_times_local, t_maxs)  # looks like: {'/p': 3, '/b': None, ...}

            return t_maxs, stutter_word_idxs
        
        # TTS: care about /p and /b
        elif client_type == 'tts':
            
            # Thresholds for relevant stutter types
            thresholds = {
                '/p': self.thresholds_tensor[0].item(),
                '/b': self.thresholds_tensor[1].item()  # convert to dict for easier access
            }

            # Estimate buffer-local regions for /p and /b
            p_start_buf, p_end_buf = self.estimate_region_local(center_ts,
                                                          window_probs['/p'],
                                                          thresholds['/p'])
            b_start_buf, b_end_buf = self.estimate_region_local(center_ts,
                                                          window_probs['/b'],
                                                          thresholds['/b'])

            # Convert to segment-local (seconds since beg_time)
            p_start_seg, p_end_seg = self.buf_to_seg_local(p_start_buf, p_end_buf)
            b_start_seg, b_end_seg = self.buf_to_seg_local(b_start_buf, b_end_buf)

            # Return regions for TTS to use with p_destutter / b_destutter
            # Times are SEGMENT-local seconds, matching what p_destutter / b_destutter expect.
            return (p_start_seg, p_end_seg), (b_start_seg, b_end_seg)

        else:
            return None

    
    def crossfade_concat(self, audio1, audio2, fade_len=FADE):
        '''Helper function to concatenate two audio segments with a crossfade to avoid clicks.
        fade_len is in seconds.'''
        
        fade_samples = int(fade_len * self.sr)
        
        audio1 = np.asarray(audio1, dtype=np.float32).copy()
        audio2 = np.asarray(audio2, dtype=np.float32).copy()

        fade_samples = int(fade_len * self.sr)
        fade_samples = max(1, min(fade_samples, len(audio1), len(audio2)))

        if len(audio1) == 0:
            return audio2
        if len(audio2) == 0:
            return audio1
        if fade_samples <= 1:
            return np.concatenate([audio1, audio2])

        fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=True, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)

        overlap = audio1[-fade_samples:] * fade_out + audio2[:fade_samples] * fade_in

        return np.concatenate([
            audio1[:-fade_samples],
            overlap,
            audio2[fade_samples:]
        ])
    

    def crossfade_concat_three(self, audio1, audio2, audio3, fade_len=FADE):
        '''Helper function to concatenate three audio segments with crossfades.
        First concatenates audio1 and audio2, then concatenates the result with audio3.'''
        temp = self.crossfade_concat(audio1, audio2, fade_len)
        return self.crossfade_concat(temp, audio3, fade_len)

    
    def p_destutter(self, audio_arr, t_start, t_end,):
        '''Destutter for /p: shorten the region but keep a small part
        Note: t_start and t_end are SEGMENT local, not audio buffer local like rest of code'''
        if t_start is None or t_end is None:
            return audio_arr

        # Get sample indices
        start_idx = max(0, int(t_start * self.sr))
        end_idx = min(len(audio_arr), int(t_end * self.sr))

        if end_idx <= start_idx:
            return audio_arr

        region = audio_arr[start_idx:end_idx]
        region_len = len(region)

        if region_len < 2:   # too short to crop
            return audio_arr

        region_dur = region_len / self.sr   # time in s

        # Target final post-crop length
        target_keep = TARGET_KEEP  # kinda redundant code but makes it easier to remember what this was for

        # If the prolongation is already short, don't touch it
        if region_dur <= target_keep:
            return audio_arr

        keep_len = int(target_keep * self.sr)
        if keep_len <= 0:
            return audio_arr

        # Keep the first target_keep seconds, delete the rest
        region_short = region[:keep_len]

        if CROSSFADE:
            out = self.crossfade_concat_three(audio_arr[:start_idx], region_short, audio_arr[end_idx:])  # new - crossfade concat

        else:
            out = np.concatenate([audio_arr[:start_idx], region_short, audio_arr[end_idx:]])  # old - hard concat
        
        # Debug prints
        if DEBUG:
            t_start_global, t_end_global = self.seg_local_to_global(t_start, t_end)
            print(f'p_destutter for t from {t_start_global} to {t_end_global} success')

        return out


    def b_destutter(self, audio_arr, t_start, t_end):
        '''Block destutter: remove most of [t_start, t_end], leaving a short pause.
        Note: t_start and t_end are SEGMENT local, not audio buffer local like rest of code'''
        if t_start is None or t_end is None:
            return audio_arr

        # Get sample indices
        start_idx = max(0, int(t_start * self.sr))
        end_idx = min(len(audio_arr), int(t_end * self.sr))

        if end_idx <= start_idx:
            return audio_arr

        region = audio_arr[start_idx:end_idx]
        region_len = len(region)
        if region_len == 0:
            return audio_arr

        # Target pause length
        pause_len = int(PAUSE * self.sr)

        # If region is already shorter than this, just leave it as-is
        if region_len <= pause_len:
            return audio_arr

        # Keep a 100 ms chunk from the middle as a small pause
        start_pause = (region_len - pause_len) // 2
        end_pause   = start_pause + pause_len
        pause = region[start_pause:end_pause]

        # audio_before + short_pause + audio_after
        if CROSSFADE:
            out = self.crossfade_concat_three(audio_arr[:start_idx], pause, audio_arr[end_idx:])  # new - crossfade concat

        else:
            out = np.concatenate([audio_arr[:start_idx], pause, audio_arr[end_idx:]])  # old - hard concat
        
        # Debug prints
        if DEBUG:
            t_start_global, t_end_global = self.seg_local_to_global(t_start, t_end)
            print(f'b_destutter for t from {t_start_global} to {t_end_global} success')

        return out

    def r_destutter(self):
        '''Destutter for /r'''
        # Just remove all stuff like s-s-s
        # Logic copied over from remove_r_stutter() in destutter_pseudo.py
        pattern = r'\b([a-zA-Z]{1,3})(?:-\1){1,}-([a-zA-Z]+)'
        self.text = re.sub(pattern, r'\2', self.text)

        # Debug prints
        if DEBUG:
            print(f'r_destutter for t from {self.beg_time} to {self.end_time} success')

    def wr_destutter(self):
        '''Destutter for /wr'''
        # Logic copied over from remove_wr_stutter() in destutter_pseudo.py
        pattern = r'\b([a-zA-Z]+)(?: \1){2,}\b'
        self.text = re.sub(pattern, r'\1', self.text)

        # Debug prints
        if DEBUG:
            print(f'wr_destutter for t from {self.beg_time} to {self.end_time} success')

    def i_destutter(self, stutter_word_idxs):
        '''Destutter for /i.
        Only removes a word if it is on the known interjection whitelist —
        this prevents real content words from being deleted when the timing
        estimate snaps to the wrong word.'''

        # Canonical interjections and filler words that should be removed
        INTERJECTION_WHITELIST = {
            'um', 'uh', 'er', 'eh', 'ah', 'oh', 'hmm', 'hm', 'mhm',
            'like', 'so', 'well', 'right', 'okay', 'ok', 'yeah', 'yes',
            'no', 'actually', 'basically', 'literally', 'you', 'know',
            'i', 'mean', 'just', 'kind', 'sort', 'thing', 'stuff',
        }

        # For debug
        if DEBUG:
            popped_words = []

        if stutter_word_idxs.get('/i', None) is not None:
            idx = stutter_word_idxs['/i']
            if idx < len(self.words):
                word = self.words[idx].lower().strip(".,!?;:'\"")
                if word in INTERJECTION_WHITELIST:
                    if DEBUG:
                        popped_words.append(self.words[idx])
                    self.words.pop(idx)
                    self.text = ' '.join(self.words)
                elif DEBUG:
                    print(f'i_destutter: skipping "{self.words[idx]}" — not in interjection whitelist')

        if DEBUG:
            print(f'i_destutter for t from {self.beg_time} to {self.end_time} success; popped {popped_words}')            

    def get_center_times_local(self, win_start, win_end):
        '''Get local center times (in seconds) for one window
        Assumes win_start and win_end are in (local) samples'''
        
        # Convert to seconds
        win_start = win_start / self.sr
        win_end = win_end / self.sr

        center = (win_start + win_end) / 2.0  # global center time

        return center
    
    def get_audio_stutter_probs(self, audio_arr):
        '''Given an audio array, returns the stutter probabilities for that audio using the StutterNet model'''

        # Kaldi fbank needs at least one 25 ms frame
        # If the slice is too short or empty, just treat it as "no stutter"
        if audio_arr is None or len(audio_arr) < 0.25*16000:
            return {
                '/p': 0.0,
                '/b': 0.0,
                '/r': 0.0,
                '/wr': 0.0,
                '/i': 0.0,
            }

        # Process audio through StutterNet model
        probs = self.stutter_model.infer(audio_arr)  # is_print is by default false

        # Make into dictionary
        probs = {'/p': probs[0],
                 '/b': probs[1],
                 '/r': probs[2],
                 '/wr': probs[3],
                 '/i': probs[4]}

        return probs  # is like {'/p': 0.7, '/b': 0.2, ...}

    def get_max_times_local(self, centers, window_probs, thresholds):
        '''For each stutter label that MATTERS, find the window index with max probability for each label.
        If that max prob >= thresholds[label], return the corresponding time.
        Otherwise return None for that label.

        centers: list of times (in seconds) for each window index
        window_probs: dict like {'/p': [p0, p1, ...], '/b': [...], ...}
        thresholds: dict like {'/r': 0.6, '/wr': 0.5, ...}; determines that stutter type matters
        Returns: {label: time or None, etc.}'''

        t_maxs = {}

        # For the probabilities of each label
        for label, probs in window_probs.items():
            if not probs or label not in thresholds.keys():  # if empty or not the stutter type that matters (e.g. /p)
                t_maxs[label] = None
                continue
            max_val = max(probs)
            max_idx = probs.index(max_val)
            if max_val >= thresholds[label]:  # if exceeds threshold
                t_maxs[label] = centers[max_idx]
            else:
                t_maxs[label] = None
        
        return t_maxs

    def estimate_word_times_local(self, segment_duration: float):
        '''Approximate (start,end) time in seconds (local) for each word proportional
        to how long that word is (num of characters).
        Returns list of (start_t, end_t) for each word.'''
        if not self.words:
            return []

        lengths = [len(word) for word in self.words]
        total_len = sum(lengths)
        t = 0.0
        times = []
        t_to_text = self.beg_time - self.t_to_buffer   # converts to audio buffer local time

        for i in range(len(lengths)):
            length = lengths[i]
            frac = length / total_len
            dur = segment_duration * frac
            start_t = t + t_to_text
            end_t = start_t + dur
            times.append((start_t, end_t))
            t = end_t

        return times
    
    def word_index_for_time_local(self, t: float, word_times):
        '''Given word_times in local seconds and a target time,
        return index of the word whose center is closest to target time.'''
        if not word_times:
            return None

        best_idx = 0
        best_dist = math.inf
        for i, (w_start, w_end) in enumerate(word_times):
            center = 0.5 * (w_start + w_end)
            dist = abs(center - t)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx
    
    def get_stutter_times_local(self, word_times, t_maxs):
        '''Given word_times in local seconds and t_maxs dict like {'/p': 12.3, '/b': None, ...},
        return dict like {'/p': word_idx or None, '/b': None, ...}'''

        stutter_word_idxs = {}
        for label, t_max in t_maxs.items():
            if t_max is not None:
                word_idx = self.word_index_for_time_local(t_max, word_times)
                stutter_word_idxs[label] = word_idx
            else:
                stutter_word_idxs[label] = None
        
        return stutter_word_idxs
    
    def estimate_region_local(self, centers, probs, thresh, min_consecutive=MIN_CONSECUTIVE):
        '''Estimates stutter time region from a time series of probabilities.
        centers: [c0, c1, ...] where c0, c1 etc. are local center times (s)
        probs:   [p0, p1, ...] matching centers
        thresh:  probability threshold
        min_consecutive: number of consecutive above-threshold windows required to
                         open a region. Prevents single-window spikes from triggering cuts.
        Returns (t_start, t_end) in buffer-local seconds, or (None, None) if nothing.'''

        if not centers or not probs:
            return None, None

        in_region = False
        cur_start = None
        cur_end = None
        consecutive_count = 0

        best_start = None
        best_end = None
        best_len = 0.0

        for i in range(len(centers)):
            t = centers[i]
            p = probs[i]

            if p >= thresh:
                consecutive_count += 1

                if consecutive_count >= min_consecutive:
                    if not in_region:
                        in_region = True
                        # Back-date start to the first window in this run
                        first_in_run = i - (consecutive_count - 1)
                        cur_start = centers[first_in_run]
                    cur_end = t

            else:
                # Allow a single-window blip before closing the region
                if i != len(centers) - 1 and probs[i + 1] >= thresh:
                    continue  # skip this one dip, don't reset consecutive_count
                else:
                    consecutive_count = 0
                    if in_region:
                        in_region = False
                        length = cur_end - cur_start
                        if length > best_len:
                            best_len = length
                            best_start, best_end = cur_start, cur_end

        # If we ended still inside a region
        if in_region and cur_start is not None and cur_end is not None:
            length = cur_end - cur_start
            if length > best_len:
                best_start, best_end = cur_start, cur_end

        return best_start, best_end
    
    def buf_to_seg_local(self, t_start_buf, t_end_buf):
        '''Convert audio-buffer local times (s) to segment-local times (s).
        Buffer-local: time measured from the start of audio_buffer (t_to_buffer).
        Segment-local: time measured from the start of this text segment (beg_time).'''
        # Guard in case no prolongation/block detected and thus t_start_buf, t_end_buf = None, None
        if t_start_buf is None or t_end_buf is None:
            return None, None
        
        # Segment start time in buffer time coords
        t_to_seg = self.beg_time - self.t_to_buffer  # seconds since buffer start

        t_start_seg = t_start_buf - t_to_seg
        t_end_seg = t_end_buf - t_to_seg

        return t_start_seg, t_end_seg
    
    def buf_local_to_global(self, t_start_buf, t_end_buf):
        '''Convert audio buffer local times to global times for print debugging'''
        # Guard
        if t_start_buf is None or t_end_buf is None:
            return None, None
        
        t_start_global = t_start_buf + self.t_to_buffer
        t_end_global = t_end_buf + self.t_to_buffer

        return t_start_global, t_end_global
    
    def seg_local_to_global(self, t_start_seg, t_end_seg):
        '''Convert text segement local times to global times for print debugging'''
        # Guard
        if t_start_seg is None or t_end_seg is None:
            return None, None
        
        t_start_global = t_start_seg + self.beg_time
        t_end_global = t_end_seg + self.beg_time

        return t_start_global, t_end_global


def calibrate_aud_thresholds(destutterer, fluent_audio_path, percentile=95, hop_sec=0.25, window_sec=3.0):
    '''Calibrate AUD_THRESH_P and AUD_THRESH_B by running StutterNet over a known
    fluent audio file and finding the percentile of /p and /b probabilities.

    The idea: on fluent speech the model should output low probabilities for /p and /b.
    Setting the threshold above the Nth percentile of those probabilities ensures we only
    fire on audio that is clearly more stutter-like than typical fluent speech.
    '''
    import librosa

    # Load audio if a path was given
    if isinstance(fluent_audio_path, str):
        audio, _ = librosa.load(fluent_audio_path, sr=destutterer.sr, mono=True)
    else:
        audio = np.asarray(fluent_audio_path, dtype=np.float32)

    sr = destutterer.sr
    hop_samples = int(hop_sec * sr)
    window_samples = int(window_sec * sr)
    chunk_len = len(audio)

    if chunk_len < hop_samples:
        raise ValueError(f'Audio is too short ({chunk_len} samples) for calibration. Need at least {hop_samples}.')

    all_p = []
    all_b = []

    for win_start in range(0, chunk_len - hop_samples, hop_samples):
        win_end = min(win_start + window_samples, chunk_len)
        segment = audio[win_start:win_end]
        probs = destutterer.get_audio_stutter_probs(segment)
        all_p.append(probs['/p'])
        all_b.append(probs['/b'])

    if not all_p:
        raise ValueError('No windows were processed. Check audio length and hop_sec.')

    all_p = np.array(all_p)
    all_b = np.array(all_b)

    thresh_p = float(np.percentile(all_p, percentile))
    thresh_b = float(np.percentile(all_b, percentile))

    print(f'\n=== Calibration results ({len(all_p)} windows, {percentile}th percentile) ===')
    print(f'/p  — min={all_p.min():.3f}  median={np.median(all_p):.3f}  95th={np.percentile(all_p,95):.3f}  max={all_p.max():.3f}')
    print(f'/b  — min={all_b.min():.3f}  median={np.median(all_b):.3f}  95th={np.percentile(all_b,95):.3f}  max={all_b.max():.3f}')
    print(f'\nRecommended thresholds (set these in destutterer.py):')
    print(f'  AUD_THRESH_P = {thresh_p:.3f}')
    print(f'  AUD_THRESH_B = {thresh_b:.3f}')
    print(f'\nIf get false positives, re-run with a higher percentile')
    print(f'If miss real stutters, try a lower percentile\n')

    return thresh_p, thresh_b