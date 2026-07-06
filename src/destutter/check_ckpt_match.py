"""
Checks whether the checkpoint (63.pt) actually matches the model architecture
currently defined by train_convlstm.yaml, key by key. A partial/mismatched load is
the leading suspect after test_model_responsiveness.py showed the model's OUTPUT is
essentially frozen regardless of dramatically different inputs (silence, full-scale
clipping, loud noise, pure tones) despite the underlying features clearly varying.

Usage: python check_checkpoint_match.py
"""

import torch
import yaml
from wenet.utils.init_model import init_model

CONFIG_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\sed\\examples\\stutter_event\\s0\\conf\\train_convlstm.yaml'
CKPT_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\exp\\convlstm_en\\63.pt'
CMVN_PATH = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\global_cmvn'

# Build the model exactly the way StutterSED.__init__ does, so architecture matches
with open(CONFIG_PATH, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)

if 'fbank_conf' in configs['dataset_conf']:
    configs['input_dim'] = configs['dataset_conf']['fbank_conf'].get('num_mel_bins', 80)
elif 'mfcc_conf' in configs['dataset_conf']:
    configs['input_dim'] = configs['dataset_conf']['mfcc_conf'].get('num_ceps', 40)
else:
    configs['input_dim'] = 80

configs['cmvn_file'] = CMVN_PATH
configs['is_json_cmvn'] = True

model = init_model(configs)
model_keys = set(model.state_dict().keys())

# Load the raw checkpoint file directly - don't go through load_checkpoint(), so we
# see exactly what's on disk regardless of how that function handles mismatches
raw = torch.load(CKPT_PATH, map_location='cpu')
# Checkpoints are sometimes the state_dict directly, sometimes wrapped in a dict
# with a 'state_dict'/'model' key - handle both
if isinstance(raw, dict) and 'state_dict' in raw:
    ckpt_state = raw['state_dict']
elif isinstance(raw, dict) and 'model' in raw and isinstance(raw['model'], dict):
    ckpt_state = raw['model']
else:
    ckpt_state = raw

ckpt_keys = set(ckpt_state.keys())

missing_from_ckpt = model_keys - ckpt_keys   # model expects these, checkpoint doesn't have them -> stay at random init
unexpected_in_ckpt = ckpt_keys - model_keys  # checkpoint has these, current model architecture doesn't use them -> silently ignored
matched = model_keys & ckpt_keys

print(f"Model expects {len(model_keys)} parameter tensors.")
print(f"Checkpoint contains {len(ckpt_keys)} parameter tensors.")
print(f"Matched (loaded correctly): {len(matched)}")
print()

if missing_from_ckpt:
    print(f"MISSING FROM CHECKPOINT ({len(missing_from_ckpt)}) - these stayed at random init, never got trained weights:")
    for k in sorted(missing_from_ckpt):
        print(f"  {k}")
    print()
else:
    print("Nothing missing from checkpoint - every parameter the model expects was found.\n")

if unexpected_in_ckpt:
    print(f"UNEXPECTED IN CHECKPOINT ({len(unexpected_in_ckpt)}) - present in the file but unused by the current architecture "
          f"(often means the checkpoint was trained under a different config/architecture than train_convlstm.yaml currently defines):")
    for k in sorted(unexpected_in_ckpt):
        print(f"  {k}")
    print()
else:
    print("Nothing unexpected in checkpoint - every key in the file is used by the current model.\n")

if missing_from_ckpt or unexpected_in_ckpt:
    print("=> Architecture mismatch confirmed. Some layers are running on random weights.")
    print("   This fully explains why the model's output doesn't respond to its input.")
else:
    print("=> Keys match exactly. The checkpoint mismatch theory is NOT the explanation -")
    print("   the frozen output must be coming from somewhere else (e.g. a shape mismatch")
    print("   that still let load_checkpoint succeed but silently reshaped/dropped data,")
    print("   or a bug in the model's forward()/decode() itself).")