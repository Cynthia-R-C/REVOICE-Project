"""
Same test as test_stutter_clips.py, but targeting ConvLSTM instead of StutterNet -
now that the double-CMVN-normalization bug is fixed, this checks whether ConvLSTM's
/p and /b actually discriminate real prolongations/blocks from fluent speech, the same
way we just tested (and disproved) for StutterNet.

clip_prolongation_small_3to6s.wav should contain "...content with this sssssssmall
group in in the villa" - a real /p-type prolongation.
clip_block_tentative_13to16s.wav should contain "just a t-t-tentative" - a real
/b-type block/repetition.

Usage: python test_stutter_clips_convlstm.py
"""

import librosa
import numpy as np
from destutterer import Destutterer

# ConvLSTM
CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_convlstm.yaml'
CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\convlstm_en\\63.pt'
CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\global_cmvn'
SR = 16000

PROLONGATION_CLIP_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\whisper_streaming\\stuttered_audio\\clip_prolongation_small_3to6s.wav'
BLOCK_CLIP_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\whisper_streaming\\stuttered_audio\\clip_block_tentative_13to16s.wav'
FLUENT_CLIP_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\src\\destutter\\fluent_sample.wav'

AUD_THRESH_P = 0.37
AUD_THRESH_B = 0.48

d = Destutterer(CONFIG_PATH, CKPT_PATH, CMVN_PATH, sr=SR)

def load_and_test(name, path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) < int(3.0 * SR):
        print(f"WARNING: {name} is shorter than 3.0s ({len(audio)/SR:.2f}s) - "
              f"pad or use a longer clip for a fair test.")
    audio = audio[:int(3.0 * SR)]  # exactly one window's worth
    probs = d.get_audio_stutter_probs(audio)
    print(f"{name:<30} /p={probs['/p']:.4f} (thresh {AUD_THRESH_P})   "
          f"/b={probs['/b']:.4f} (thresh {AUD_THRESH_B})")
    return probs

print(f"{'Clip':<30} {'':<35}")
print('-' * 75)
p_prolong = load_and_test('prolongation clip (real /p)', PROLONGATION_CLIP_PATH)
p_block = load_and_test('block clip (real /b)', BLOCK_CLIP_PATH)
if FLUENT_CLIP_PATH:
    p_fluent = load_and_test('fluent clip (should be low)', FLUENT_CLIP_PATH)