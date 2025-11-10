#!/usr/bin/env python3
# encoding: utf-8
# Modified for mfcc instead of fbank

import sys
import argparse
import json
import codecs
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader


class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''

    def __init__(self, feat_dim, resample_rate, feats_type, fbank_conf=None, mfcc_conf=None):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate
        self.feats_type = feats_type
        self.fbank_conf = fbank_conf or {}
        self.mfcc_conf = mfcc_conf or {}

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            value = item[1].strip().split(",")
            assert len(value) == 3 or len(value) == 1
            wav_path = value[0]
            sample_rate = torchaudio.info(wav_path).sample_rate
            resample_rate = sample_rate
            if len(value) == 3:
                start_frame = int(float(value[1]) * sample_rate)
                end_frame = int(float(value[2]) * sample_rate)
                waveform, sample_rate = torchaudio.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(item[1])

            waveform = waveform * (1 << 15)
            if self.resample_rate != 0 and self.resample_rate != sample_rate:
                resample_rate = self.resample_rate
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate)(waveform)

            # change: select feature type
            if self.feats_type == 'mfcc':
                mat = kaldi.mfcc(
                    waveform,
                    num_mel_bins=self.mfcc_conf.get('num_mel_bins', 40),
                    num_ceps=self.mfcc_conf.get('num_ceps', 40),
                    frame_length=self.mfcc_conf.get('frame_length', 25),
                    frame_shift=self.mfcc_conf.get('frame_shift', 10),
                    low_freq=self.mfcc_conf.get('low_freq', 20),
                    high_freq=self.mfcc_conf.get('high_freq', -400),
                    dither=self.mfcc_conf.get('dither', 0.1),
                    sample_frequency=resample_rate
                )
            else:
                mat = kaldi.fbank(
                    waveform,
                    num_mel_bins=self.fbank_conf.get('num_mel_bins', 80),
                    dither=self.fbank_conf.get('dither', 0.0),
                    energy_floor=0.0,
                    sample_frequency=resample_rate
                )
            # end change

            mean_stat += torch.sum(mat, axis=0)
            var_stat += torch.sum(torch.square(mat), axis=0)
            number += mat.shape[0]
        return number, mean_stat, var_stat


class AudioDataset(Dataset):
    def __init__(self, data_file):
        self.items = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split()
                self.items.append((arr[0], arr[1]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract CMVN stats')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--train_config', default='', help='training yaml conf')
    parser.add_argument('--in_scp', default=None, help='wav.scp file')
    parser.add_argument('--out_cmvn', default='global_cmvn', help='output file')
    parser.add_argument('--log_interval', type=int, default=1000)
    args = parser.parse_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    dataset_conf = configs['dataset_conf']
    feats_type = dataset_conf.get('feats_type', 'fbank')
    fbank_conf = dataset_conf.get('fbank_conf', {})
    mfcc_conf = dataset_conf.get('mfcc_conf', {})

    if feats_type == 'mfcc':
        feat_dim = mfcc_conf.get('num_ceps', 40)
    else:
        feat_dim = fbank_conf.get('num_mel_bins', 80)

    resample_rate = 0
    if 'resample_conf' in dataset_conf:
        resample_rate = dataset_conf['resample_conf'].get('resample_rate', 0)
        print(f'using resample, new rate = {resample_rate}')

    collate_func = CollateFunc(feat_dim, resample_rate, feats_type, fbank_conf, mfcc_conf)
    dataset = AudioDataset(args.in_scp)

    data_loader = DataLoader(dataset,
                             batch_size=20,
                             shuffle=True,
                             num_workers=args.num_workers,
                             collate_fn=collate_func)

    with torch.no_grad():
        all_number = 0
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += 20
            if wav_number % args.log_interval == 0:
                print(f'processed {wav_number} wavs, {all_number} frames',
                      file=sys.stderr, flush=True)

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }

    with open(args.out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))
