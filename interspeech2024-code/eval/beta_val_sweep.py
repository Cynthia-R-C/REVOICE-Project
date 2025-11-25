import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Constants

ROOT = Path('C:/Users/crc24/Documents/VS_Code_Python_Folder/ScienceFair2025')

# Paths to scripts
THRESH_SCRIPT = ROOT / 'interspeech2024-code/sed/wenet/wenet/bin/compute_thresholds.py'
INFER_SCRIPT   = ROOT / 'interspeech2024-code/sed/wenet/wenet/bin/infer_sed.py'

# Paths to config / info files
TRAIN_CONFIG   = ROOT / 'interspeech2024-code/sed/examples/stutter_event/s0/conf/train_stutternet.yaml'
TUNING_CONFIG  = ROOT / 'interspeech2024-code/eval/tuning_config.json'
CMVN_PATH      = ROOT / 'interspeech2024-code/data/train/global_cmvn'
DATASET_LIST   = ROOT / 'interspeech2024-code/data/test/infer_data.list'
CHECKPOINT     = ROOT / 'interspeech2024-code/exp/stutternet_en/36.pt'

# Output paths
THRESH_OUTPUT  = ROOT / 'interspeech2024-code/eval/thresholds.pt'
INFER_RESULT_DIR = ROOT / 'interspeech2024-code/eval'
OUT_DIR = INFER_RESULT_DIR / 'beta_sweep_results'

# beta sweep configuration
BETA_MIN = 0.0
BETA_MAX = 5.0
BETA_STEP = 0.05

# Stutter types
STUTTER_LABELS = ['/p', '/b', '/r', '/wr', '/i']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEBUG = True


# Helper: run command in subprocess and ensure safety

def run_cmd(cmd_list):
    '''Run a command and return (success, output).
    Code basically taken from Medium article by Doug Creates.'''
    try:
        p = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        return stdout.decode(), stderr.decode()
    
    except Exception as e:
        return '', str(e)


# Helper: update tuning_config.json

def update_tuning_config(beta_vals):
    '''Updates the tuning_config.json file with new f-beta values.'''
    with open(TUNING_CONFIG, 'r') as f:
        tuning_config = json.load(f)

    for lbl, val in beta_vals.items():
        tuning_config['f_beta'][lbl] = float(val)

    with open(TUNING_CONFIG, 'w') as f:
        json.dump(tuning_config, f, indent=4)

    if DEBUG:
        with open(TUNING_CONFIG, 'r') as f:
            tuning_config = json.load(f)
        print(f"Using betas for this run: {tuning_config['f_beta']}")  # check if beta vals are actually changing


# Helper: parse precision/recall lines from infer result

def parse_infer_results():
    '''Returns lists of prec and rec values for each class'''
    results_file = INFER_RESULT_DIR / 'infer_sed_results.txt'
    if not results_file.exists():
        return None

    prec = None
    rec = None

    with open(results_file, 'r') as f:
        for line in f:
            if line.startswith('Rec:'):
                parts = line.strip().split()[1:]  # skip "Rec:"
                rec = [float(x) for x in parts]

            if line.startswith('Prec:'):
                parts = line.strip().split()[1:]  # skip "Prec:"
                prec = [float(x) for x in parts]

            # Once both found, stop (prevents reading the random baseline block)
            if prec is not None and rec is not None:
                return prec, rec

    return None

def sanitize(label):
    '''To prevent illegal file names'''
    return label.replace('/', '')


# Main beta sweep

def main():
    # Prepare storage for curves
    beta_values = np.arange(BETA_MIN, BETA_MAX + BETA_STEP, BETA_STEP)

    # 5 classes × list of values across beta sweep
    prec_curves = {label: [] for label in STUTTER_LABELS}
    rec_curves    = {label: [] for label in STUTTER_LABELS}

    # Sweep over beta values
    for beta in beta_values:
        betas = {
            '/p':   float(beta),
            '/b':   float(beta),
            '/r':   float(beta),
            '/wr':  float(beta),
            '/i': float(beta)
        }

        print(f'\n==== Testing beta = {beta:.2f} ====')

        # 1. Update tuning config
        update_tuning_config(betas)

        # 2. Compute thresholds
        cmd_thresh = [
            'python', str(THRESH_SCRIPT),
            '--config', str(TRAIN_CONFIG),
            '--tuning_config', str(TUNING_CONFIG),
            '--checkpoint', str(CHECKPOINT),
            '--cmvn', str(CMVN_PATH),
            '--dataset', str(DATASET_LIST),
            '--gpu', '0',
            '--output', str(THRESH_OUTPUT),
            '--graph', 'False'  # no point in graphing every time
        ]
        stdout, stderr = run_cmd(cmd_thresh)
        if stderr and ('Traceback' in stderr or 'Error' in stderr):
            print('compute_threshold.py failed:', stderr)
            continue

        if DEBUG:
            threshold = torch.load(THRESH_OUTPUT).to(device)  # load computed thresholds
            print(f'Thresholds: {threshold}')

        # 3. Run inference
        cmd_infer = [
            'python', str(INFER_SCRIPT),
            '--config', str(TRAIN_CONFIG),
            '--tuning_config', str(TUNING_CONFIG),
            '--dataset', str(DATASET_LIST),
            '--checkpoint', str(CHECKPOINT),
            '--cmvn', str(CMVN_PATH),
            '--threshold', str(THRESH_OUTPUT),
            '--result_dir', str(INFER_RESULT_DIR),
            '--gpu', '0',
            '--data_type', 'shard',
            '--batch_size', '16'
        ]
        stdout, stderr = run_cmd(cmd_infer)
        if stderr and ('Traceback' in stderr or 'Error' in stderr):
            print('infer_sed.py failed:', stderr)
            continue

        # 4. Parse inference results
        parsed = parse_infer_results()
        if parsed is None:
            print('Could not parse infer_sed_results.txt')
            continue

        prec, rec = parsed
        if DEBUG:
            print(f'Precision: {prec}')
            print(f'Recall: {rec}')

        # Store values
        for i, label in enumerate(STUTTER_LABELS):
            prec_curves[label].append(prec[i])
            rec_curves[label].append(rec[i])

    # 5. Plotting
    OUT_DIR.mkdir(exist_ok=True)

    # Beta vs Precision
    for label in STUTTER_LABELS:
        safe_label = sanitize(label)
        plt.figure(figsize=(8,5))
        plt.plot(beta_values, prec_curves[label], label=f'{label} precision')
        plt.title(f'Precision vs Beta ({label})')
        plt.xlabel('Beta')
        plt.ylabel('Precision (%)')
        plt.grid(True)
        plt.savefig(OUT_DIR / f'precision_{safe_label}.png')
        plt.close()

    # Beta vs Recall
    for label in STUTTER_LABELS:
        safe_label = sanitize(label)
        plt.figure(figsize=(8,5))
        plt.plot(beta_values, rec_curves[label], label=f'{label} recall')
        plt.title(f'Recall vs Beta ({label})')
        plt.xlabel('Beta')
        plt.ylabel('Recall (%)')
        plt.grid(True)
        plt.savefig(OUT_DIR / f'recall_{safe_label}.png')
        plt.close()

    print('\nBeta sweep completed.')
    print(f'Graphs saved in: {OUT_DIR}')


if __name__ == '__main__':
    main()
