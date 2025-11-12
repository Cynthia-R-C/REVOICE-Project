# Quick program to regenerate inference datalist with updated paths (so base dir is ScienceFair2025)
# Cynthia Chen 11/10/2025

# Not gonna do command prompt for this because I think I'll only use this once.

import os
from pathlib import Path

datalist_path = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\data.list'  # use absolute path because tired...

new_base_dir = 'C:\\Users\\crc24\\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\data\\train\\infer_data.list'  # new base dir for wav files

with open(datalist_path, 'r', encoding='utf8') as orig_f:
    with open(new_base_dir, 'w', encoding='utf8') as new_f:
        for line in orig_f:  # for each tar path
            new_line = 'interspeech2024-code/' + line  # prepend new base dir
            new_f.write(new_line)  # write to new file

print(f"Regenerated inference datalist at: {new_base_dir}")