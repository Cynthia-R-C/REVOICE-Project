# Purpose: write the json file with f-beta values for tuning
# Cynthia Chen 11/20/2025

import json

P_BETA = 1.0
B_BETA = 1.0
R_BETA = 1.0
WR_BETA = 1.0
INT_BETA = 1.0

data = {
    "f_beta": {
        "p": P_BETA,
        "b": B_BETA,
        "r": R_BETA,
        "wr": WR_BETA,
        "int": INT_BETA
    }
}

with open('tuning_config.json', 'w') as f:
    json.dump(data, f, indent=4)