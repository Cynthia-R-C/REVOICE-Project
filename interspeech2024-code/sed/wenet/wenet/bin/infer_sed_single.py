#!/usr/bin/env python3
# encoding: utf-8

# Cynthia Chen 12/3/2025 (class implementation)
# Logic largely copied from infer_sed.py
# Modified for single inference + class implementation

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torchaudio
import yaml
import numpy as np

from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.cmvn import load_cmvn

def get_args():
    parser = argparse.ArgumentParser(description='infer single audio with your model')
    parser.add_argument('--config', required=True, help='config file (e.g. train_stutternet.yaml)')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model (.pt)')
    parser.add_argument('--cmvn', required=True, help='path to CMVN file')
    parser.add_argument('--threshold', required=False, help='path to threshold file')  # threshold file will serve as the toggle for whether to just compute the probabilities or to compute the existence of the label
    # e.g. if threshold file exists, do like /p: 1, /b: 0 etc.
    # if threshold file doesn't exist, do like /p: 0.75, /b: 0.02, etc.
    parser.add_argument('--audio', required=True, help='input audio; may be file or arr')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--sr',
                        type=int,
                        default=16000,
                        help='sampling rate for audio')
    parser.add_argument('--aud_type', required=False, default='arr', choices=['arr','file'])  # toggle for using audio arr or wav file
    parser.add_argument('--over_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    args = parser.parse_args()
    print(args)
    return args


class StutterSED:

    def __init__(self, config_path, ckpt_path, cmvn_path, thresh_path=None, gpu=-1, sr=16000, aud_type='arr', over_config=[]):  # same as parser arguments
        '''Initialize StutterSED class and run global calculations; e.g. ones that don't depend on audio
        audio may be either a wav path or a numpy arr depending on last toggle'''

        # Instance variables
        self.gpu = gpu
        self.device = 'cuda' if gpu>=0 and torch.cuda.is_available() else 'cpu'
        self.sr = sr
        self.aud_type = aud_type
        self.override_config = over_config
        self.threshold = None  # intialize as None so later guard code works

        # Load configs
        with open(config_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        if len(over_config) > 0:
            configs = override_config(configs, over_config)

        # disable augmentation and shuffling for inference
        self.test_conf = copy.deepcopy(configs['dataset_conf'])
        self.test_conf['speed_perturb'] = False
        self.test_conf['spec_aug'] = False
        self.test_conf['shuffle'] = False
        self.test_conf['sort'] = False
        if 'fbank_conf' in self.test_conf:
            self.test_conf['fbank_conf']['dither'] = 0.0
        elif 'mfcc_conf' in self.test_conf:
            self.test_conf['mfcc_conf']['dither'] = 0.0

        # Determine input dimension from feature type
        if 'fbank_conf' in configs['dataset_conf']:
            configs['input_dim'] = configs['dataset_conf']['fbank_conf'].get('num_mel_bins', 80)
        elif 'mfcc_conf' in configs['dataset_conf']:
            configs['input_dim'] = configs['dataset_conf']['mfcc_conf'].get('num_ceps', 40)
        else:
            configs['input_dim'] = 80  # safe default

        # Initialize model
        configs['cmvn_file'] = cmvn_path
        configs['is_json_cmvn'] = True
        self.model = init_model(configs)
        load_checkpoint(self.model, ckpt_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Apply thresholds only if specified
        self.thresh_path = thresh_path   # for deciding whether to apply thresholds in infer()
        if thresh_path:
            # threshold = torch.Tensor([[0.42, 0.35, 0.37, 0.37, 0.4]]).to(device)  # old
            self.threshold = torch.load(self.thresh_path).to(self.device)  # load computed thresholds

        # Load CMVN
        mean, istd = load_cmvn(cmvn_path, True)
        self.mean = torch.tensor(mean).to(self.device).float()
        self.istd = torch.tensor(istd).to(self.device).float()


    def infer(self, audio, is_print=False):
        '''Run inference process for one audio input
        audio may be a file (must be toggled) or an array (default)'''

        # Logging - taken from infer_sed.py
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(message)s')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)


        # ======== Load and preprocess audio ======== #

        # if audio is wav path
        if self.aud_type == 'file':
            wav, sr = torchaudio.load(audio)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
        
        else:   # aud_type is arr, so it's already a numpy array at the right sr
            wav_np = np.asarray(audio, dtype=np.float32)  # just in case do this
            wav = torch.from_numpy(wav_np).unsqueeze(0)  # convert to tensor

        feats_type = self.test_conf.get('feats_type', 'fbank')
        if feats_type == 'fbank':
            fbank_conf = self.test_conf.get('fbank_conf', {})
            feats = torchaudio.compliance.kaldi.fbank(
                wav,
                num_mel_bins=fbank_conf.get('num_mel_bins', 80),
                frame_length=fbank_conf.get('frame_length', 25),
                frame_shift=fbank_conf.get('frame_shift', 10),
                dither=fbank_conf.get('dither', 0.0),
                sample_frequency=self.sr
            )
        else:  # MFCC
            mfcc_conf = self.test_conf.get('mfcc_conf', {})
            feats = torchaudio.compliance.kaldi.mfcc(
                wav,
                num_mel_bins=mfcc_conf.get('num_mel_bins', 40),
                num_ceps=mfcc_conf.get('num_ceps', 40),
                frame_length=mfcc_conf.get('frame_length', 25),
                frame_shift=mfcc_conf.get('frame_shift', 10),
                low_freq=mfcc_conf.get('low_freq', 20),
                high_freq=mfcc_conf.get('high_freq', -400),
                dither=mfcc_conf.get('dither', 0.1),
                sample_frequency=self.sr
            )

        # Added fix for dtype mismatch
        feats = feats.float()

        feats = feats.to(self.device)
        feats = (feats - self.mean) * self.istd
        feats = feats.unsqueeze(0)
        feats_lengths = torch.tensor([feats.size(1)]).to(self.device)


        # ======== Inference ======== #

        with torch.no_grad():
            logits = self.model.decode(feats, feats_lengths)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            if self.thresh_path:  # if we wanna apply the thresholds
                results = (probs > self.threshold.cpu().numpy().flatten()).astype(int)
            else:
                results = probs


        # ======== Results ======== #

        if is_print:   # print readable results to terminal

            labels = ['/p', '/b', '/r', '/wr', '/i']

            if self.threshold is not None:
                print(f'threshold {self.threshold}')
            else:
                print('threshold: None (using raw probabilities)')

            print("\n=== StutterNet Results ===")
            for i, label in enumerate(labels):
                print(f"{label}: {results[i]} (prob={probs[i]:.4f})")  # if thresholds not intended to be used, will just output probabilities for both results and probs

        print("\nSingle inference success.")

        return results


# ======== Testing ======== #
if __name__ == '__main__':
    args = get_args()
    stutter_model = StutterSED(args.config, args.checkpoint, args.cmvn, args.threshold, args.gpu, args.sr, args.aud_type, args.over_config)
    results = stutter_model.infer(args.audio, is_print=True)  # no need to print results; infer() already automatically doess
