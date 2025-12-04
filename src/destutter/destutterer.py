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

        # Non-segment-specific instance variables are initialized here
        self.config = config_path
        self.checkpoint = ckpt_path
        self.cmvn = cmvn_path
        self.sr = sr # sampling rate
        self.device = device

        # Load class StutterSED
        self.stutter_model = StutterSED(self.config, self.checkpoint, self.cmvn, gpu=0 if self.device=='cuda' else -1)

    def get_text(self):
        '''Returns current text'''
        return self.text

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
        hop = 0.05 * self.sr   # could increase this if latency is too high
        window_size = 3.0 * self.sr  # 3s window; window size in samples
        loc_start = (self.beg_time - self.t_to_buffer) * self.sr
        loc_end = (self.end_time - self.t_to_buffer) * self.sr

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
        THRESH_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\eval\\thresholds.pt'
        thresholds_tensor = torch.load(THRESH_PATH).to(self.device)  # load computed thresholds

        # STT: care about /r, /wr, /i
        if client_type == 'stt':   # note to self: can later modify this to only calculate values for needed stt stutter types to reduce latency

            # Thresholds for relevant stutter types
            thresholds = {'/r': thresholds_tensor[2].item(),
                          '/wr': thresholds_tensor[3].item(),
                          '/i': thresholds_tensor[4].item() }  # convert to dict for easier access

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
                '/p': thresholds_tensor[0].item(),
                '/b': thresholds_tensor[1].item()  # convert to dict for easier access
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


    # Out of date - kept for reference
    def destutter(self):
        '''OUT OF DATE: Main destuttering logic
        Only for reference - in actual whisper_online_server.py, 
        destuttering is done in parts based on stt/tts type.
        Mainly for keeping track of logic'''
        # So far only updated for STT

        ## Simple: SOUND REP ##
        self.r_destutter()

        # Sliding 3s window over audio corresponding to text segment
        hop = 0.05 * self.sr   # could increase this if latency is too high
        window_size = 3.0 * self.sr  # 3s window; window size in samples
        loc_start = (self.beg_time - self.t_to_buffer) * self.sr
        loc_end = (self.end_time - self.t_to_buffer) * self.sr

        window_probs = {'/p': [], '/b': [], '/r': [], '/wr': [], '/i': []}
        center_ts = []  # local center times of each window (in seconds)

        # Guard: if segment is shorter than window (highly unlikely), throw an error for now, will handle later depending on if this actually happens
        if loc_end - loc_start <= window_size:
            raise ValueError(f'Audio segment shorter than window size for destuttering; audio segment len = {loc_end - loc_start}, window size = {window_size}')

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
        THRESH_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\eval\\thresholds.pt'
        thresholds = torch.load(THRESH_PATH).to(self.device)  # load computed thresholds
        t_maxs = self.get_max_times_local(center_ts, window_probs, thresholds)  # looks like: {'/p': 12.3, '/b': None, ...}
        # So by now via t_maxs it is has been determined whether a stutter type is actually present or not

        ## Medium: WORD REP ##

        # Only run this if word rep detected in model(?) might not be necessary, could just run this without model
        if t_maxs['/wr'] is not None:
            self.wr_destutter()

        # Approximate local start/end times for each word in the text segment
        segment_duration = self.end_time - self.beg_time
        word_times_local = self.estimate_word_times_local(segment_duration)

        # if any stutter detected, find the corresponding word index(s)
        stutter_word_idxs = self.get_stutter_times_local(word_times_local, t_maxs)  # looks like: {'/p': 3, '/b': None, ...}


        ## Complex: INTERJECTIONS ##
        self.i_destutter(stutter_word_idxs)


        ## Remaning: PROLONGATIONS, BLOCKS ##
        # To be implemented for TTS later

        return self.text   # will add in audio for this later
    
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
        target_keep = 0.12  # 120 ms, tune this by ear (e.g. 0.10–0.15)

        # If the prolongation is already short, don't touch it
        if region_dur <= target_keep:
            return audio_arr

        keep_len = int(target_keep * self.sr)
        if keep_len <= 0:
            return audio_arr

        # Keep the first target_keep seconds, delete the rest
        region_short = region[:keep_len]

        out = np.concatenate([audio_arr[:start_idx], region_short, audio_arr[end_idx:]])
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

        # Target pause length: ~100 ms
        pause_len = int(0.10 * self.sr)

        # If region is already shorter than this, just leave it as-is
        if region_len <= pause_len:
            return audio_arr

        # Keep a 100 ms chunk from the middle as a small pause
        start_pause = (region_len - pause_len) // 2
        end_pause   = start_pause + pause_len
        pause = region[start_pause:end_pause]

        # audio_before + short_pause + audio_after
        out = np.concatenate([audio_arr[:start_idx], pause, audio_arr[end_idx:]])
        return out

    def r_destutter(self):
        '''Destutter for /r'''
        # Just remove all stuff like s-s-s
        # Logic copied over from remove_r_stutter() in destutter_pseudo.py
        pattern = r'\b([a-zA-Z]{1,3})(?:-\1){1,}-([a-zA-Z]+)'
        self.text = re.sub(pattern, r'\2', self.text)

    def wr_destutter(self):
        '''Destutter for /wr'''
        # Logic copied over from remove_wr_stutter() in destutter_pseudo.py
        pattern = r'\b([a-zA-Z]+)(?: \1){2,}\b'
        self.text = re.sub(pattern, r'\1', self.text)

    def i_destutter(self, stutter_word_idxs):
        '''Destutter for /i'''
        # Remove interjection word if it's detected
        if stutter_word_idxs['/i'] is not None:
            idx = stutter_word_idxs['/i']
            self.words.pop(idx)  # remove the interjection word
            self.text = ' '.join(self.words)

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
    
    def estimate_region_local(self, centers, probs, thresh):
        '''Estimates stutter time region from a time series of probabilities
        centers: [c0, c1, ...] where c0, c1 etc. are local center times (s)
        probs:   [p0, p1, ...] matching centers
        thresh: probability threshold
        Returns (t_start, t_end) in segment-local seconds, or (None, None) if nothing.'''

        if not centers or not probs:
            return None, None

        in_region = False
        cur_start = None
        cur_end = None

        best_start = None
        best_end = None
        best_len = 0.0

        # For every time step
        for i in range(len(centers)):
            t = centers[i]
            p = probs[i]

            # Stutter detected at this time step
            if p >= thresh:
                if not in_region:
                    in_region = True
                    cur_start = t  # approx region start
                cur_end = t        # keep extending
            
            # Stutter no longer detected
            else:
                # Check if it's a blip
                if i != len(centers) - 1 and probs[i + 1] >= thresh:  # next step has stutter
                    continue  # ignore this blip
                else:  # not a blip, end of region
                    if in_region:
                        in_region = False
                        length = cur_end - cur_start
                        if length > best_len:
                            best_len = length
                            best_start, best_end = cur_start, cur_end

        # If we ended still inside a region
        if in_region:
            length = cur_end - cur_start
            if length > best_len:
                best_start, best_end = cur_start, cur_end

        return best_start, best_end
    
    def buf_to_seg_local(self, t_start_buf, t_end_buf):
        '''Convert audio-buffer local times (s) to segment-local times (s).
        Buffer-local: time measured from the start of audio_buffer (t_to_buffer).
        Segment-local: time measured from the start of this text segment (beg_time).
        '''
        # Guard in case no prolongation/block detected and thus t_start_buf, t_end_buf = None, None
        if t_start_buf is None or t_end_buf is None:
            return None, None
        
        # Segement start time in buffer time coords
        t_to_seg = self.beg_time - self.t_to_buffer  # seconds since buffer start

        t_start_seg = t_start_buf - t_to_seg
        t_end_seg   = t_end_buf   - t_to_seg

        return t_start_seg, t_end_seg
