# Purpose: write the json file with f-beta values for tuning
# Cynthia Chen 11/20/2025

import json

P_BETA = 0.7  # precision
B_BETA = 0.7  # precision
R_BETA = 1.5  # recall
WR_BETA = 2.2 # recall
INT_BETA = 0.8  # precision OR equal

data = {
    "f_beta": {
        "p": P_BETA,
        "b": B_BETA,
        "r": R_BETA,
        "wr": WR_BETA,
        "int": INT_BETA
    }
}

with open('C:\\Users\\crc24\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\eval\\tuning_config.json', 'w') as f:
    json.dump(data, f, indent=4)
print('Successfully wrote tuning_config.json.')