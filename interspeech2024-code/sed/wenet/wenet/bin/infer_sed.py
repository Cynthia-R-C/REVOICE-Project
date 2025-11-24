# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for StutterNet inference - Cynthia Chen 11/10/2025

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset_sed import Dataset
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model

# from sklearn.metrics import fbeta_score  # never mind - we need to compute weighted f1 manually


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('tuning_config', help='tuning config json with fbeta vals')
    parser.add_argument('--dataset', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--cmvn', required=True, help='path to CMVN file')  # added
    parser.add_argument('--threshold', required=True, help='path to threshold file')  # added
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    args = parser.parse_args()
    print(args)
    return args

def calc_hit_hyp_ref(results, target):
    '''Calculates positives - hit, hyp, ref'''
    # hit = true positives
    # hyp = predicted positives
    # ref = actual positives
    hit = torch.logical_and(results, target).int().sum(0)
    hyp = results.sum(0)
    ref = target.sum(0)
    return hit, hyp, ref

def calc_rec_prec_f1(hit, hyp, ref):
    '''Calculates recall, precision, f1 score'''
    # rec = recall
    # prec = precision
    # f1 = f1 score

    def to_string(t, do_round=True):
        return '\t'.join([str(round(r * 100, 2)) if do_round else str(r) for r in t.tolist()])

    rec = hit / ref   # true positives / actual positives
    prec = hit / hyp  # true positives / predicted positives
    f1 = 2 * rec * prec / (rec + prec)
    out = ''
    out += '\t/p\t/b\t/r\t/wr\t/i\n'
    out += 'Rec:\t'+to_string(rec)+'\n'
    out += 'Prec:\t'+to_string(prec)+'\n'
    out += 'F1:\t'+to_string(f1)+'\n'
    out += 'hit:\t'+to_string(hit, False)+'\n'
    out += 'hyp:\t'+to_string(hyp, False)+'\n'
    out += 'ref:\t'+to_string(ref, False)+'\n'
    return out

def calc_rec_prec_weighted_f1(hit, hyp, ref, y_true, y_probs, beta):
    '''Calculates weighted f1 score with beta value from tuning config'''
    rec = hit / ref   # true positives / actual positives
    prec = hit / hyp  # true positives / predicted positives
    f1 = fbeta_score(y_true, y_probs, beta=beta, zero_division=0)

    def to_string(t, do_round=True):
        return '\t'.join([str(round(r * 100, 2)) if do_round else str(r) for r in t.tolist()])

    out = ''
    out += '\t/p\t/b\t/r\t/wr\t/i\n'
    out += 'Rec:\t'+to_string(rec)+'\n'
    out += 'Prec:\t'+to_string(prec)+'\n'
    out += 'F1:\t'+to_string(f1)+'\n'
    out += 'hit:\t'+to_string(hit, False)+'\n'
    out += 'hyp:\t'+to_string(hyp, False)+'\n'
    out += 'ref:\t'+to_string(ref, False)+'\n'
    return out

def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Read config file
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # Create and use test config

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
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    test_dataset = Dataset(args.data_type,
                           args.dataset,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Determine input dimension from feature type
    if 'mfcc_conf' in configs['dataset_conf']:
        configs['input_dim'] = configs['dataset_conf']['mfcc_conf'].get('num_ceps', 40)
    elif 'fbank_conf' in configs['dataset_conf']:
        configs['input_dim'] = configs['dataset_conf']['fbank_conf'].get('num_mel_bins', 80)
    else:
        configs['input_dim'] = 80  # safe default

    # Init sed model from configs
    configs['cmvn_file'] = args.cmvn  # add hardcode
    configs['is_json_cmvn'] = True  # add hardcode
    model = init_model(configs)

    # Load model checkpoint and thresholds
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    # threshold = torch.Tensor([[0.42, 0.35, 0.37, 0.37, 0.4]]).to(device)  # old thresholds
    threshold = torch.load(args.threshold).to(device)  # load computed thresholds
    print(f'threshold {threshold}')

    # TODO(Dinghao Zhou): Support RNN-T related decoding
    # TODO(Lv Xiang): Support k2 related decoding
    # TODO(Kaixun Huang): Support context graph
    f = open(os.path.join(args.result_dir, 'infer_sed_results.txt'), 'w')

    # Begin inference
    with torch.no_grad():

        # Metrics for actual model use
        hit_all = torch.Tensor([0, 0, 0, 0, 0]).to(device)
        hyp_all = torch.Tensor([0, 0, 0, 0, 0]).to(device)
        ref_all = torch.Tensor([0, 0, 0, 0, 0]).to(device)

        for batch_idx, batch in enumerate(test_data_loader):
            # target = unprocessed y_true labels
            keys, feats, target_actual, feats_lengths, target_lengths = batch
            feats = feats.to(device)

            target_actual = target_actual.to(device)
            labels_actual = target_actual.int()  # what is this shape?

            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)

            results_actual = model.decode(      # results = probs
                feats,
                feats_lengths)
            results_actual = (results_actual > threshold).int()

            hit, hyp, ref = calc_hit_hyp_ref(results_actual, target_actual)
            hit_all += hit
            hyp_all += hyp
            ref_all += ref
            #logging.info(f'batch: {batch_idx}')

        # Do again but for random baseline
        hit_rand = torch.Tensor([0, 0, 0, 0, 0]).to(device)
        hyp_rand = torch.Tensor([0, 0, 0, 0, 0]).to(device)
        ref_rand = torch.Tensor([0, 0, 0, 0, 0]).to(device)
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target_rand, feats_lengths, target_lengths = batch
            target_rand = target_rand.to(device)
            labels_rand = target_rand.int()
            results_rand = torch.randint(0, 2, target_rand.shape, device=device)
            hit, hyp, ref = calc_hit_hyp_ref(results_rand, target_rand)
            hit_rand += hit
            hyp_rand += hyp
            ref_rand += ref
            #logging.info(f'batch: {batch_idx}')

    # Get beta values from config

    stats_all = calc_rec_prec_weighted_f1(hit_all, hyp_all, ref_all, labels_actual.cpu().numpy(), results_actual.cpu().numpy(), beta=1.0)
    # stats_all = calc_rec_prec_f1(hit_all, hyp_all, ref_all)
    stats_rand = calc_rec_prec_weighted_f1(hit_rand, hyp_rand, ref_rand, labels_rand.cpu().numpy(), results_rand.cpu().numpy())
    print(stats_all)
    print(stats_rand)

    f.write(stats_all)
    f.write(stats_rand)
    f.close()

if __name__ == '__main__':
    main()