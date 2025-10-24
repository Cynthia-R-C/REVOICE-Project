# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

#!/usr/bin/env python3

# Modified for SED training by removing DeepSpeed (Cynthia Chen 10/20/2025)

import argparse
import os
import sys
import time
import yaml
import torch
import torch.distributed as dist

from wenet.utils.checkpoint import load_checkpoint, save_checkpoint, load_trained_modules
from wenet.utils.executor import Executor
from wenet.utils.scheduler import WarmupLR, NoamHoldAnnealing
from wenet.utils.config import override_config

# ###
# # Ensure local SED 'wenet' is importable regardless of environment
# from pathlib import Path
# _THIS_FILE = Path(__file__).resolve()
# _SED_ROOT = _THIS_FILE.parents[2]    # .../sed/wenet/bin -> parents[2] = .../sed
# if str(_SED_ROOT) not in sys.path:
#     sys.path.insert(0, str(_SED_ROOT))
# ###


def get_args():
    """Simplified CLI argument parser (no DeepSpeed)"""
    parser = argparse.ArgumentParser(description='Train a SED model (DeepSpeed-free)')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type', default='raw', choices=['raw', 'shard'])
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir', default='tensorboard')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--fp16_grad_sync', action='store_true', default=False)
    parser.add_argument('--cmvn', default=None)
    parser.add_argument('--override_config', action='append', default=[])
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='gloo',
                        choices=['nccl', 'gloo'])
    parser.add_argument('--ddp.init_method', dest='init_method', default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()


def main():
    args = get_args()

    # distributed setup (still works fine single-GPU)
    distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                                world_size=world_size, rank=rank)

    with open(args.config, 'r', encoding='utf8') as f:
        configs = yaml.safe_load(f)
    configs = override_config(configs, args.override_config)
    configs['is_distributed'] = distributed
    configs['is_deepspeed'] = False  # forcibly disabled

    # basic setup
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model + executor
    executor = Executor()
    model, optimizer, scheduler = executor.build_model(configs, device=device)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, optimizer, scheduler)

    # training loop
    start_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    print(f"[{start_time}] Start training on {device} (DeepSpeed disabled)")

    executor.train(model=model,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   train_data=args.train_data,
                   cv_data=args.cv_data,
                   model_dir=model_dir,
                   cmvn=args.cmvn,
                   use_amp=args.use_amp,
                   rank=rank,
                   world_size=world_size)

    if rank == 0:
        save_checkpoint(model, os.path.join(model_dir, 'final.pt'))
        print("✅ Training complete. Model saved to", model_dir)


if __name__ == '__main__':
    sys.exit(main())
