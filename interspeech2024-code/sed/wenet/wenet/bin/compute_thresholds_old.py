# Code for computing the classification thresholds for StutterNet model
# Cynthia Chen 11/10/2025

import argparse
import yaml
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from wenet.utils.init_model import init_model
from wenet.utils.cmvn import load_cmvn
from wenet.utils.file_utils import read_lists
import torchaudio

def get_args():
    '''Get needed arguments from command line'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cmvn", required=True)
    parser.add_argument("--dev_list", required=True,
                        help="Path to data list for validation")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--output", default="threshold.pt")  # change this default later so it saves in exp/stutternet_en
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    # Load config, model, and CMVN
    with open(args.config, 'r') as fin:   # config
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    mean, istd = load_cmvn(args.cmvn, is_json_cmvn=True)  # cmvn
    model = init_model(configs)  # initialize model
    checkpoint = torch.load(args.checkpoint, map_location="cpu")  # load checkpoint
    model.load_state_dict(checkpoint["model"])  # checkpoint is a dictionary file, so load the model part this way
    model.eval()   # eval mode

    # Get true labels and predicted probabilities on dev set
    all_probs = []
    all_labels = []
    with open(args.dev_list, "r") as f:
        lines = f.readlines()

    # For each audio file in dev set
    for line in tqdm(lines):
        parts = line.strip().split()
        wav = parts[0]   # this gets the path to the .tar shard file for the wav
        labels = np.array(list(map(int, parts[1:])))
        wav_data, sr = torchaudio.load(wav)
        if sr != args.sr:
            wav_data = torchaudio.functional.resample(wav_data, sr, args.sr)
        feats = torchaudio.compliance.kaldi.mfcc(
            wav_data, num_ceps=40, num_mel_bins=40, sample_frequency=args.sr
        ).float()
        feats = (feats - mean) * istd
        feats = feats.unsqueeze(0)
        with torch.no_grad():
            logits = model.decode(feats, torch.tensor([feats.shape[1]]))
        probs = torch.sigmoid(logits).numpy().squeeze()
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    thresholds = []
    for i in range(all_probs.shape[1]):
        p, r, t = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        f1 = 2 * p * r / (p + r + 1e-9)
        best_t = t[np.argmax(f1)]
        thresholds.append(best_t)

    thresholds = torch.tensor([thresholds])
    torch.save(thresholds, args.output)
    print("Saved thresholds:", thresholds)

if __name__ == "__main__":
    main()
