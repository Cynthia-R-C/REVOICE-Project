"""
Two checks to isolate why StutterSED's output doesn't respond to its input, now that
we've ruled out a checkpoint/architecture mismatch (all 20 keys matched exactly):

1. STATE CARRYOVER CHECK: build a completely FRESH model instance for each test
   input (rather than reusing the same instance across calls, like
   test_model_responsiveness.py did). If a fresh, isolated instance run on JUST
   silence gives a meaningfully different result than a separate fresh instance run
   on JUST loud noise, that proves some kind of internal state (e.g. an LSTM hidden
   state stored as a persistent attribute instead of being reset each call) was
   carrying over between calls and dominating the result in the earlier test.

2. BATCHNORM CHECK: print any BatchNorm layers' running_mean / running_var. If these
   don't roughly match the scale of the CMVN-normalized features actually being fed
   in (mean around -4 to -6, std around 0.8-1.5, per the feats stats we've printed),
   BatchNorm would renormalize everything into a similar, saturated range regardless
   of the real input differences - which would fully explain a frozen output even
   with a correctly-loaded checkpoint.

Usage: python check_state_and_batchnorm.py
"""

import numpy as np
import torch
from destutterer import Destutterer

CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_convlstm.yaml'
CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\convlstm_en\\63.pt'
CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\global_cmvn'
SR = 16000
WINDOW_SEC = 3.0
window_samples = int(WINDOW_SEC * SR)

test_inputs = {
    'all_zeros (silence)': np.zeros(window_samples, dtype=np.float32),
    'random_noise_loud': (np.random.randn(window_samples).astype(np.float32) * 0.5),
    'sine_2000hz_loud': (0.8 * np.sin(2 * np.pi * 2000 * np.arange(window_samples) / SR)).astype(np.float32),
}

print("=" * 75)
print("CHECK 1: fresh model instance per input (rules out hidden-state carryover)")
print("=" * 75)
for name, arr in test_inputs.items():
    d = Destutterer(CONFIG_PATH, CKPT_PATH, CMVN_PATH, sr=SR)  # brand new instance, no shared state possible
    probs = d.get_audio_stutter_probs(arr)
    print(f"{name:<25} /p={probs['/p']:.6f}  /b={probs['/b']:.6f}  /r={probs['/r']:.6f}  "
          f"/wr={probs['/wr']:.6f}  /i={probs['/i']:.6f}")

print()
print("If these numbers now differ meaningfully between inputs (unlike the shared-")
print("instance test), the earlier result was caused by state carrying over between")
print("calls. If they're still frozen even with a fresh instance per call, state")
print("carryover is ruled out and the problem is stateless (e.g. BatchNorm below).")
print()

print("=" * 75)
print("CHECK 2: BatchNorm layer running stats")
print("=" * 75)
d = Destutterer(CONFIG_PATH, CKPT_PATH, CMVN_PATH, sr=SR)
found_bn = False
for name, module in d.stutter_model.model.named_modules():
    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        found_bn = True
        rm = module.running_mean
        rv = module.running_var
        print(f"{name}:")
        print(f"  running_mean: min={rm.min().item():.4f} max={rm.max().item():.4f} mean={rm.mean().item():.4f}")
        print(f"  running_var:  min={rv.min().item():.4f} max={rv.max().item():.4f} mean={rv.mean().item():.4f}")

if not found_bn:
    print("No BatchNorm layers found in the model.")
    print("(If the model uses LayerNorm/GroupNorm instead, those don't carry running")
    print(" stats the same way and wouldn't cause this specific failure mode - the")
    print(" freeze would have to be coming from somewhere else in the architecture,")
    print(" e.g. a sigmoid/activation saturating due to a large learned bias, or a")
    print(" bug in how hidden state is threaded through the ConvLSTM's forward pass.)")
else:
    print()
    print("Compare these running_mean/running_var ranges against the feats stats we")
    print("printed earlier (mean around -4 to -6, std around 0.8-1.5). If BatchNorm's")
    print("stored stats are wildly different from that scale, it will renormalize any")
    print("real input into a similarly saturated range, explaining the frozen output.")