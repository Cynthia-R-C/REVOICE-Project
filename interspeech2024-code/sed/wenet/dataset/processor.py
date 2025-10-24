# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==================== processor.py (Windows-compatible patch) ====================
# This patched version keeps SOX functionality for Linux
# and automatically switches to a safe SoundFile backend on Windows.

import os
import io
import math
import torch
import random
import tarfile
import platform
import torchaudio
import numpy as np
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

# ==================== Windows/Linux audio backend handling ====================
if platform.system() == "Windows":
    print("[Info] Windows detected: switching torchaudio backend to 'soundfile'")
    torchaudio.set_audio_backend("soundfile")

    # Create harmless stub for SOX utilities
    class _NoSox:
        def set_buffer_size(self, *args, **kwargs):
            pass

    torchaudio.utils.sox_utils = _NoSox()

else:
    # Original Linux path
    import torchaudio.utils.sox_utils as sox_ext
    sox_ext.set_buffer_size(16500)

# ==================== Core Dataset Processor ====================

class TarFileAndGroup(object):
    def __init__(self, tar_file):
        self.tar_file = tar_file
        self.tar_obj = None
        self.members = []

    def open(self):
        self.tar_obj = tarfile.open(self.tar_file, "r")
        self.members = self.tar_obj.getmembers()

    def close(self):
        if self.tar_obj is not None:
            self.tar_obj.close()


def load_wav(path, resample=16000):
    """Load audio file, resample if necessary."""
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != resample:
        waveform = Resample(sample_rate, resample)(waveform)
    return waveform, resample


def compute_fbank(waveform, sample_rate, num_mel_bins=80):
    """Compute filterbank features."""
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate,
        use_energy=False,
    )
    return mat


def speed_perturb(waveform, sample_rate, speed=1.0):
    """Apply speed perturbation. SOX on Linux, resample fallback on Windows."""
    if abs(speed - 1.0) < 1e-6:
        return waveform
    if platform.system() != "Windows":
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
    else:
        # Simple tempo approximation for Windows
        new_rate = int(sample_rate * speed)
        wav = Resample(orig_freq=sample_rate, new_freq=new_rate)(waveform)
    return wav


class AudioDataset(Dataset):
    """Example dataset class that reads audio tar shards."""

    def __init__(self, tar_list, num_mel_bins=80, resample=16000):
        self.tar_list = tar_list
        self.num_mel_bins = num_mel_bins
        self.resample = resample
        self.shards = [TarFileAndGroup(t) for t in tar_list]

    def __len__(self):
        return len(self.shards)

    def __getitem__(self, idx):
        tar = self.shards[idx]
        tar.open()
        wav_files = [m for m in tar.members if m.name.endswith(".wav")]
        txt_files = [m for m in tar.members if m.name.endswith(".txt")]

        samples = []
        for wf, tf in zip(wav_files, txt_files):
            audio = tar.tar_obj.extractfile(wf).read()
            txt = tar.tar_obj.extractfile(tf).read().decode("utf-8").strip()

            # Load waveform from bytes
            waveform, sr = torchaudio.load(io.BytesIO(audio))
            if sr != self.resample:
                waveform = Resample(sr, self.resample)(waveform)

            # Optional speed perturbation
            spd = random.choice([0.9, 1.0, 1.1])
            waveform = speed_perturb(waveform, self.resample, spd)

            # Compute fbank features
            mat = compute_fbank(waveform, self.resample, self.num_mel_bins)

            samples.append((mat, txt))

        tar.close()
        return samples


if __name__ == "__main__":
    # Simple test
    print("[Info] Processor self-test: loading dummy audio file")
    tmp_wav = "test.wav"
    if os.path.exists(tmp_wav):
        wav, sr = load_wav(tmp_wav)
        fbank = compute_fbank(wav, sr)
        print("✅ Loaded", tmp_wav, "shape:", fbank.shape)
    else:
        print("[Info] No test.wav found — skipping test.")
