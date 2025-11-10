#!/usr/bin/env python3
# encoding: utf-8

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torchaudio
import yaml

from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.cmvn import load_cmvn


def get_args():
    parser = argparse.ArgumentParser(description='recognize single audio with your model')
    parser.add_argument('--config', required=True, help='config file (e.g. train_stutternet.yaml)')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model (.pt)')
    parser.add_argument('--cmvn', required=True, help='path to CMVN file')
    parser.add_argument('--wav', required=True, help='input wav file path')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--sr',
                        type=int,
                        default=16000,
                        help='sampling rate for audio')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # disable augmentation and shuffling for inference
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0

    # Determine input dimension from feature type
    if 'mfcc_conf' in configs['dataset_conf']:
        configs['input_dim'] = configs['dataset_conf']['mfcc_conf'].get('num_ceps', 40)
    elif 'fbank_conf' in configs['dataset_conf']:
        configs['input_dim'] = configs['dataset_conf']['fbank_conf'].get('num_mel_bins', 80)
    else:
        configs['input_dim'] = 80  # safe default

    # Initialize model
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True
    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    threshold = torch.Tensor([[0.42, 0.35, 0.37, 0.37, 0.4]]).to(device)
    print(f'threshold {threshold}')

    # Load CMVN
    mean, istd = load_cmvn(args.cmvn, True)
    mean = torch.tensor(mean).to(device).float()
    istd = torch.tensor(istd).to(device).float()

    # Load and preprocess WAV
    wav, sr = torchaudio.load(args.wav)
    if sr != args.sr:
        wav = torchaudio.functional.resample(wav, sr, args.sr)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    feats_type = test_conf.get('feats_type', 'mfcc')
    if feats_type == 'fbank':
        fbank_conf = test_conf.get('fbank_conf', {})
        feats = torchaudio.compliance.kaldi.fbank(
            wav,
            num_mel_bins=fbank_conf.get('num_mel_bins', 80),
            frame_length=fbank_conf.get('frame_length', 25),
            frame_shift=fbank_conf.get('frame_shift', 10),
            dither=fbank_conf.get('dither', 0.0),
            sample_frequency=args.sr
        )
    else:  # MFCC
        mfcc_conf = test_conf.get('mfcc_conf', {})
        feats = torchaudio.compliance.kaldi.mfcc(
            wav,
            num_mel_bins=mfcc_conf.get('num_mel_bins', 40),
            num_ceps=mfcc_conf.get('num_ceps', 40),
            frame_length=mfcc_conf.get('frame_length', 25),
            frame_shift=mfcc_conf.get('frame_shift', 10),
            low_freq=mfcc_conf.get('low_freq', 20),
            high_freq=mfcc_conf.get('high_freq', -400),
            dither=mfcc_conf.get('dither', 0.1),
            sample_frequency=args.sr
        )

    # Added fix for dtype mismatch
    feats = feats.float()

    feats = feats.to(device)
    feats = (feats - mean) * istd
    feats = feats.unsqueeze(0)
    feats_lengths = torch.tensor([feats.size(1)]).to(device)

    with torch.no_grad():
        logits = model.decode(feats, feats_lengths)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        results = (probs > threshold.cpu().numpy().flatten()).astype(int)


    labels = ['/p', '/b', '/r', '/wr', '/i']

    print("\n=== StutterNet Results ===")
    for i, label in enumerate(labels):
        print(f"{label}: {results[i]} (prob={probs[i]:.4f})")

    print("\nDone.")


if __name__ == '__main__':
    main()
