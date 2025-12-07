# Purpose: write the json file with f-beta values for tuning
# Cynthia Chen 11/20/2025

import json

# Target f-beta scores
# P: rec > 25, prec > 60
# B: rec > 10, prec > 50
# R: rec > 75, prec > 15
# Wr: rec > 75, prec > 10
# Int: rec > 40, prec > 80

P_BETA = 0.47  # precision
B_BETA = 0.36  # precision
R_BETA = 2.5  # recall
WR_BETA = 2.5 # recall
INT_BETA = 0.5  # precision

data = {
    'f_beta': {
        '/p': P_BETA,
        '/b': B_BETA,
        '/r': R_BETA,
        '/wr': WR_BETA,
        '/i': INT_BETA
    }
}

with open('C:\\Users\\crc24\Documents\\VS_Code_Python_Folder\\ScienceFair2025\\interspeech2024-code\\eval\\tuning_config.json', 'w') as f:
    json.dump(data, f, indent=4)
print('Successfully wrote tuning_config.json.')