# Quick file to confirm that the CMVN was indeed computed from the MFCC 40 dims

import json

with open('C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\global_cmvn', 'r') as f:
    cmvn = json.load(f)

print("mean_stat length:", len(cmvn['mean_stat']))  # if 40 means it's MFCC correct, if 80 then uh oh