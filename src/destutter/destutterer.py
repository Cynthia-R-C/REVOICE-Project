# Destutterer class to handle destuttering using StutterNet model
# Implemented in whisper_online_server.py
# Cynthia Chen 11/30/2025

import torch
import math
import numpy as np
import re

class Destutterer:
    '''Class to handle destuttering using StutterNet model'''

    def __init__(self, t_to_buffer, audio_buffer, beg_time, end_time, text, sr=16000):
        '''Initialize all needed parameters'''
        self.t_to_buffer = t_to_buffer  # time between global stream start and audio buffer start
        self.audio_buffer = audio_buffer  # current audio buffer - a numpy array of samples
        self.beg_time = beg_time  # global start time of text (s)
        self.end_time = end_time  # global end time of text (s)
        self.text = text  # text segment
        self.words = self.text.split(' ')  # list of words in text segment
        self.sr = sr # sampling rate

    def destutter(self):
        '''Main destuttering logic'''
        # So far only updated for STT

        ## Simple: SOUND REP ##
        self.r_destutter()

        # Sliding 3s window over audio corresponding to text segment
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hop = 0.05 * self.sr   # could increase this if latency is too high
        window_size = 3.0 * self.sr  # 3s window; window size in samples
        loc_start = (self.beg_time - self.t_to_buffer) * self.sr
        loc_end = (self.end_time - self.t_to_buffer) * self.sr
        window_probs = {'/p': [], '/b': [], '/r': [], '/wr': [], '/i': []}

        # Getting the stutter probabilities for each window
        for win_start in range(loc_start, loc_end - window_size, hop):
            audio_segment = self.audio_buffer[int(win_start):int(win_start + window_size)]  # get audio segment corresponding to text
            stutter_probs = self.get_audio_stutter_probs(audio_segment)  # get stutter probabilities for this audio segment
            # sttuter_probs is like {'/p': 0.7, '/b': 0.2, ...}
            
            # Store stutter probabilities for this window
            for label in window_probs.keys():
                window_probs[label].append(stutter_probs[label])  # append the prob for this window
                # window_probs is like {'/p': [p0, p1, ...], '/b': [...], ...}
        
        # Find the times of max probs for each label
        THRESH_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\eval\\thresholds.pt'
        thresholds = torch.load(THRESH_PATH).to(device)  # load computed thresholds
        t_maxs = self.get_max_times_local(window_probs, thresholds)  # looks like: {'/p': 12.3, '/b': None, ...}
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
            self.text = ' '.join(words)
    
    def get_audio_stutter_probs(self, audio_arr):  # finish this later
        '''Given an audio array, returns the stutter probabilities for that audio using the StutterNet model'''
        # Convert audio array to torch tensor
        audio_tensor = torch.tensor(audio_arr).float().to(device)

        # Process audio through StutterNet model
        with torch.no_grad():
            probs = stutter_model(audio_tensor.unsqueeze(0))  # add batch dimension

        return probs.squeeze(0).cpu().numpy()  # remove batch dimension and convert to numpy

    def get_max_times_local(window_probs, thresholds):
        '''For each stutter label, find the window index with max probability for each label.
        If that max prob >= thresholds[label], return the corresponding time.
        Otherwise return None for that label.

        centers: list of times (in seconds) for each window index
        window_probs: dict like {'/p': [p0, p1, ...], '/b': [...], ...}
        thresholds: dict like {'/p': 0.6, '/b': 0.5, ...}
        Returns: {label: time or None, etc.}'''

        t_maxs = {}

        # For the probabilities of each label
        for label, probs in window_probs.items():
            if not probs:  # if empty
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

        for i in range(len(lengths)):
            length = lengths[i]
            frac = length / total_len
            dur = segment_duration * frac
            start_t = t
            end_t = t + dur
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