# Code for computing the classification thresholds for StutterNet model
# Cynthia Chen 11/11/2025

import argparse
import copy
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

from wenet.dataset.dataset_sed import Dataset
from torch.utils.data import DataLoader
from wenet.utils.init_model import init_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.config import override_config

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='training config yaml')
    p.add_argument('--checkpoint', required=True, help='model checkpoint .pt')
    p.add_argument('--cmvn', required=True, help='path to CMVN file (json or kaldi)')
    p.add_argument('--dev_data', required=True, help='dev shard list (e.g., data/dev/data.list)')
    p.add_argument('--data_type', default='shard', choices=['shard', 'raw'])
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--gpu', type=int, default=-1)
    p.add_argument('--override_config', action='append', default=[], help='YAML overrides')
    p.add_argument('--output', default='thresholds.pt')
    p.add_argument('--graph', default='False', choices=['True', 'False'], help='whether to use graph mode')
    p.add_argument('--graph_output_dir', required=False, help='directory to save PR curve graphs if graph mode is True')
    args = p.parse_args()

    return args

def graph_PR_curves(args, all_labels, all_probs, num_classes):
    '''Generate and save PR curves for each class'''
    class_dict = {0: 'Prolongation', 1: 'Block', 2: 'Sound Rep', 3: 'Word Rep', 4: 'Interjection'}
    for c in range(num_classes):
        y_true = all_labels[:, c].astype(np.int32)
        y_score = all_probs[:, c]
        prec, rec, thr = precision_recall_curve(y_true, y_score)

        plt.figure()
        plt.plot(rec, prec, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {class_dict[c]}')
        plt.grid()
        plt.savefig(os.path.join(args.graph_output_dir, f'pr_curve_class_{c}.png'))
        plt.close()

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load configs
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # Build a test config mirroring infer_sed.py
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 200
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    if 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = 'static'
    test_conf['batch_conf']['batch_size'] = args.batch_size

    # Dataset / Loader over shard list
    dev_dataset = Dataset(args.data_type, args.dev_data, test_conf, partition=False)
    dev_loader = DataLoader(dev_dataset, batch_size=None, num_workers=0)

    # Infer input_dim the same way as infer_sed.py
    if 'mfcc_conf' in configs['dataset_conf']:
        configs['input_dim'] = configs['dataset_conf']['mfcc_conf'].get('num_ceps', 40)
    elif 'fbank_conf' in configs['dataset_conf']:
        configs['input_dim'] = configs['dataset_conf']['fbank_conf'].get('num_mel_bins', 80)
    else:
        configs['input_dim'] = 80

    # Ensure CMVN path and format are present for init_model
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True

    # Initialize model
    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    # Run inference
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Dev inference"):
            # Dataset returns: keys, feats, target, feats_lengths, target_lengths
            # Following code is basically taken from infer_sed.py and lightly modified
            _, feats, target, feats_lengths, _ = batch
            feats = feats.to(device).float()
            feats_lengths = feats_lengths.to(device)
            target = target.to(device).int()

            logits = model.decode(feats, feats_lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = target.cpu().numpy()

            all_probs.append(probs)  # shape (batch_size, num_classes)
            all_labels.append(labels)

    # Produces new lists so that shape is (num_samples, num_classes) and there aren't different sublists for each different batch
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Per-class F1-optimal thresholds via PR curve
    num_classes = all_probs.shape[1]
    thresholds = []
    for c in range(num_classes):  # for each stutter type
        y_true = all_labels[:, c].astype(np.int32)
        y_score = all_probs[:, c]
        # precision_recall_curve returns precision, recall, thresholds with len(thr)=len(precision)-1
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        if thr.size == 0:
            # degenerate case: pick 0.5
            thresholds.append(0.5)
            continue
        f1 = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-9)  # manually compute f1s, since scikit's f1 computer requires a threshold argument; this gives an array of f1s
        best_idx = np.argmax(f1)
        thresholds.append(float(thr[best_idx]))  # get best threshold

    thresholds = torch.tensor(thresholds, dtype=torch.float32)
    torch.save(thresholds, args.output)
    print("Saved thresholds:", thresholds.tolist(), "->", args.output)

    # Graph PR curves if specified
    if args.graph == 'True':
        if args.graph_output_dir is None:
            raise ValueError("graph_output_dir must be specified if graph mode is True")
        if not os.path.exists(args.graph_output_dir):
            os.makedirs(args.graph_output_dir)
        graph_PR_curves(args, all_labels, all_probs, num_classes)

if __name__ == "__main__":
    main()
