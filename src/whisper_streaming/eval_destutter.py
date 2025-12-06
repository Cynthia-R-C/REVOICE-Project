# Evaluate destuttering effectiveness against a fluent reference text.
# Cynthia Chen 12/6/2025

# Usage (example):

# python evaluate_destutter.py --ref english_patient.txt --raw transcript_raw.txt --destut transcript_destut.txt

# This will print:
# - WER for raw and destuttered transcripts
# - Number of stutter candidates (insertions) before/after
# - Fraction of stutter candidates removed

import argparse
import re
from typing import Tuple, Dict

from jiwer import wer, compute_measures


def load_text(path: str) -> str:
    '''Load a text file into a single string.'''
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def normalize(text: str) -> str:
    '''
    Simple normalization:
    - lowercase
    - strip punctuation
    - collapse whitespace
    '''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def count_stutter_candidates(ref_text: str, hyp_text: str) -> int:
    '''Count “stutter-like” tokens based on insertions relative to the reference.

    For now, stutter candidates = number of insertion tokens in the hypothesis
    with respect to the reference after simple normalization.

    Might later refine this to:
      - only count certain filler tokens ('uh', 'um', etc.),
      - or detect immediate repeats in the hypothesis.
    '''
    ref_norm = normalize(ref_text)
    hyp_norm = normalize(hyp_text)

    measures = compute_measures(ref_norm, hyp_norm)
    # jiwer returns integer counts for 'insertions', 'deletions', 'substitutions', 'hits'
    insertions = measures.get('insertions', 0)
    return insertions


def evaluate_destuttering(ref_path, raw_path, destut_path):
    '''Compute WERs and stutter-candidate removal rate.
    Outputs a dictionary of dict[str] = float'''

    # Load texts
    ref_txt = load_text(ref_path)
    raw_txt = load_text(raw_path)
    destut_txt = load_text(destut_path)

    # WER before/after (normalized)
    ref_norm = normalize(ref_txt)
    raw_norm = normalize(raw_txt)
    destut_norm = normalize(destut_txt)

    raw_wer = wer(ref_norm, raw_norm)
    destut_wer = wer(ref_norm, destut_norm)

    # Stutter candidates before/after
    raw_stutters = count_stutter_candidates(ref_txt, raw_txt)
    destut_stutters = count_stutter_candidates(ref_txt, destut_txt)

    if raw_stutters > 0:
        removal_rate = (raw_stutters - destut_stutters) / raw_stutters
    else:
        removal_rate = 0.0

    # Print a small report
    print('=== Destutter Evaluation ===')
    print(f'Reference:          {ref_path}')
    print(f'Raw transcript:     {raw_path}')
    print(f'Destut transcript:  {destut_path}')
    print()
    print(f'WER (raw):          {raw_wer:.3f}')
    print(f'WER (destut):       {destut_wer:.3f}')
    print()
    print(f'Stutter candidates (raw):     {raw_stutters}')
    print(f'Stutter candidates (destut):  {destut_stutters}')
    print(f'Destutter removal rate:       {removal_rate:.3f}')

    return {
        'wer_raw': raw_wer,
        'wer_destut': destut_wer,
        'stutters_raw': raw_stutters,
        'stutters_destut': destut_stutters,
        'removal_rate': removal_rate,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate destuttering against a fluent reference text.'
    )
    parser.add_argument(
        '--ref',
        type=str,
        required=True,
        help='Path to reference fluent text (e.g. english_patient.txt).',
    )
    parser.add_argument(
        '--raw',
        type=str,
        required=True,
        help='Path to raw (non-destuttered) transcript.',
    )
    parser.add_argument(
        '--destut',
        type=str,
        required=True,
        help='Path to destuttered transcript.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_destuttering(args.ref, args.raw, args.destut)
