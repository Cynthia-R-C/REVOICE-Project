"""
Standalone sanity check: does StutterSED's output change AT ALL based on input?

Run this directly (not through the server) to isolate whether the model itself is
responsive to its input, independent of anything in the streaming pipeline, buffer,
or CMVN normalization path we've been debugging.

Usage: adjust CONFIG_PATH / CKPT_PATH / CMVN_PATH below to match your setup, then:
    python test_model_responsiveness.py
"""

import numpy as np
from destutterer import Destutterer

# --- these match the paths already defined in whisper_online_server.py ---
CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_convlstm.yaml'
CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\convlstm_en\\63.pt'
CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\global_cmvn'
SR = 16000
WINDOW_SEC = 3.0

d = Destutterer(CONFIG_PATH, CKPT_PATH, CMVN_PATH, sr=SR, device='cpu')

window_samples = int(WINDOW_SEC * SR)

test_inputs = {
    'all_zeros (silence)': np.zeros(window_samples, dtype=np.float32),
    'all_ones (max clip)': np.ones(window_samples, dtype=np.float32),
    'random_noise_loud': (np.random.randn(window_samples).astype(np.float32) * 0.5),
    'random_noise_quiet': (np.random.randn(window_samples).astype(np.float32) * 0.01),
    'sine_440hz': (np.sin(2 * np.pi * 440 * np.arange(window_samples) / SR)).astype(np.float32),
    'sine_2000hz_loud': (0.8 * np.sin(2 * np.pi * 2000 * np.arange(window_samples) / SR)).astype(np.float32),
}

print(f"{'Input':<25} {'/p':>8} {'/b':>8} {'/r':>8} {'/wr':>8} {'/i':>8}")
print('-' * 75)
for name, arr in test_inputs.items():
    probs = d.get_audio_stutter_probs(arr)
    print(f"{name:<25} {probs['/p']:>8.6f} {probs['/b']:>8.6f} {probs['/r']:>8.6f} {probs['/wr']:>8.6f} {probs['/i']:>8.6f}")

print()
print("If every row above shows the same numbers (to several decimal places) even for")
print("silence vs. loud noise vs. pure tones, the model is not responding to input at")
print("all - this is a model/checkpoint problem, not anything in the streaming pipeline,")
print("buffer, or CMVN normalization we've been debugging so far.")
print()
print("If the numbers DO differ meaningfully across these obviously-different inputs,")
print("the model is responsive in general, and the issue is specific to real speech")
print("audio (e.g. an actual normalization/distribution mismatch between real speech")
print("and whatever this test model was trained/calibrated on).")