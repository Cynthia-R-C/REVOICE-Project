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

from __future__ import print_function

import argparse
import copy
import datetime
import json
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml

# Personal debug mode
DEBUG = True

# patched deepseed
_DS_AVAILABLE = False
try:
    import deepspeed  # type: ignore
    _DS_AVAILABLE = True
except Exception:
    deepspeed = None  # type: ignore

# Lazy-import DS helpers inside guarded blocks (avoid hard import at module load)
# from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
# from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from wenet.dataset.dataset_sed import Dataset
from wenet.utils.checkpoint import (load_checkpoint, save_checkpoint,
                                    load_trained_modules)
from wenet.utils.executor import Executor
from wenet.utils.scheduler import WarmupLR, NoamHoldAnnealing
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=None,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument("--enc_init_mods",
                        default="encoder.",
                        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
                        help="List of encoder modules \
                        to initialize ,separated by a comma")
    parser.add_argument('--lfmmi_dir',
                        default='',
                        required=False,
                        help='LF-MMI dir')

    # Begin deepspeed related config (patched: optional)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')

    # DeepSpeed automatically adds '--deepspeed' and '--deepspeed_config'
    if _DS_AVAILABLE:
        parser = deepspeed.add_config_arguments(parser)  # type: ignore
    else:
        # Provide compatible placeholders so args.deepspeed logic remains intact.
        parser.add_argument('--deepspeed', action='store_true', default=False)
        parser.add_argument('--deepspeed_config', type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    if args.deepspeed:
        if not _DS_AVAILABLE:
            print("[Info] DeepSpeed requested but not available — falling back to standard PyTorch training.")
            args.deepspeed = False
        else:
            with open(args.deepspeed_config, 'r') as fin:
                ds_configs = json.load(fin)
            if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
                configs["ds_dtype"] = "fp16"
            elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
                configs["ds_dtype"] = "bf16"
            else:
                configs["ds_dtype"] = "fp32"

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    distributed = world_size > 1
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=world_size,
                                rank=rank)
    elif args.deepspeed and _DS_AVAILABLE:
        deepspeed.init_distributed(dist_backend=args.dist_backend,  # type: ignore
                                   init_method=args.init_method,
                                   world_size=world_size,
                                   rank=rank)
    else:
        print("[Info] DeepSpeed not available — using standard PyTorch training.")

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['spec_trim'] = False
    cv_conf['shuffle'] = False

    # DeepSpeed notes (kept, but only enforced when DS actually used)
    if args.deepspeed and _DS_AVAILABLE:
        assert train_conf['batch_conf']['batch_type'] == "static"
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        configs['accum_grad'] = ds_configs["gradient_accumulation_steps"]
        
    if DEBUG:
        print("\n\n\n[Personal debug] Preparing datasets...\n\n\n")

    train_dataset = Dataset(args.data_type, args.train_data, train_conf, True)
    cv_dataset = Dataset(args.data_type, args.cv_data, cv_conf, partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    
    if DEBUG:
        print("\n\n\n[Personal debug] Datasets ready.\n\n\n")

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        if DEBUG:
            print(f"\n\n[Personal debug] Using fbank, input_dim: {input_dim}\n\n")
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
        if DEBUG:
            print(f"\n\n[Personal debug] Using MFCC, input_dim: {input_dim}\n\n")

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True
    configs['lfmmi_dir'] = args.lfmmi_dir

    if DEBUG:
        print("\n\n\n[Personal debug] Set configs.\n\n\n")

    if rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    if DEBUG:
        print("\n[Personal debug] Written configs. Initializing model...\n")

    # Init asr model from configs
    model = init_model(configs)
    if local_rank == 0:
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        print('the number of model params: {:,d}'.format(num_params))

    if DEBUG:
        print("\n[Personal debug] Model initialized. Trying to export scripted model.\n")

    # Try to export scripted model
    if rank == 0:
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(args.model_dir, 'init.zip'))

    executor = Executor()

    # If specify checkpoint, load some info from checkpoint
    if DEBUG:
        print("\n[Personal debug] Loading checkpoint (if any)...\n")
    
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif args.enc_init is not None:
        logging.info('load pretrained encoders: {}'.format(args.enc_init))
        infos = load_trained_modules(model, args)
    else:
        infos = {}

    if DEBUG:
        print("\n[Personal debug] Checkpoint loaded (if any). Preparing training loop...\n")
    
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None

    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        device = torch.device("cuda")
        if args.fp16_grad_sync:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default as comm_hooks,
            )
            model.register_comm_hook(
                state=None, hook=comm_hooks.fp16_compress_hook
            )
    elif args.deepspeed and _DS_AVAILABLE:  # deepspeed
        # import DS helpers lazily
        from deepspeed.runtime.zero.stage_1_and_2 import (  # type: ignore
            estimate_zero2_model_states_mem_needs_all_live)
        from deepspeed.runtime.zero.stage3 import (  # type: ignore
            estimate_zero3_model_states_mem_needs_all_live)
        from deepspeed.utils.zero_to_fp32 import (  # type: ignore
            convert_zero_checkpoint_to_fp32_state_dict)

        if rank == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model, num_gpus_per_node=world_size, num_nodes=1)
            logging.info("Estimating model states memory needs (zero3)...")
            estimate_zero3_model_states_mem_needs_all_live(
                model, num_gpus_per_node=world_size, num_nodes=1)
        device = None     # Init device later
        # Initialize DeepSpeed engine
        model, optimizer, _, scheduler = deepspeed.initialize(  # type: ignore
            args=args, model=model, model_parameters=model.parameters())
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    # Optimizer & scheduler (original logic)
    if configs['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    elif configs['optim'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **configs['optim_conf'])
    else:
        raise ValueError("unknown optimizer: " + configs['optim'])

    if configs['scheduler'] == 'warmuplr':
        scheduler_type = WarmupLR
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    elif configs['scheduler'] == 'NoamHoldAnnealing':
        scheduler_type = NoamHoldAnnealing
        scheduler = NoamHoldAnnealing(optimizer, **configs['scheduler_conf'])
    else:
        raise ValueError("unknown scheduler: " + configs['scheduler'])

    # If DeepSpeed is used AND ds_config defines optimizer/scheduler, DS will override them.
    if args.deepspeed and _DS_AVAILABLE:
        # If ds_config specifies optimizer/scheduler, can pass None here.
        # Current call already initialized DS above; keep objects as-is.
        pass

    final_epoch = None
    configs['rank'] = rank
    configs['is_distributed'] = distributed   # pytorch native ddp
    configs['is_deepspeed'] = (args.deepspeed and _DS_AVAILABLE)
    configs['use_amp'] = args.use_amp

    if args.deepspeed and _DS_AVAILABLE and start_epoch == 0:
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict  # type: ignore
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir, tag='init')  # type: ignore
            if args.save_states == "model_only" and rank == 0:
                convert_zero_checkpoint_to_fp32_state_dict(
                    model_dir, "{}/init.pt".format(model_dir), tag='init')
                os.system("rm -rf {}/{}".format(model_dir, "init"))
    elif (not args.deepspeed) and start_epoch == 0 and rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    if DEBUG:
        print("\n\n[Personal debug] Training loop prepared. Starting training...\n\n")

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        device_eff = model.local_rank if (args.deepspeed and _DS_AVAILABLE) else device

        if DEBUG:
            print(f"\n\n\n[Personal debug] Starting epoch {epoch} and running executor.train...\n\n\n")
        executor.train(model, optimizer, scheduler, train_data_loader, device_eff,
                       writer, configs, scaler)
        
        if DEBUG:
            print(f"\n\n\n[Personal debug] Running executor.cv to calculate loss...\n\n\n")
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device_eff,
                                                configs)
        cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        infos = {
            'epoch': epoch, 'lr': lr, 'cv_loss': cv_loss, 'step': executor.step,
            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
        if args.deepspeed and _DS_AVAILABLE:
            from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict  # type: ignore
            with torch.no_grad():
                model.save_checkpoint(save_dir=model_dir,  # type: ignore
                                      tag='{}'.format(epoch),
                                      client_state=infos)
                if args.save_states == "model_only" and rank == 0:
                    convert_zero_checkpoint_to_fp32_state_dict(
                        model_dir, "{}/{}.pt".format(model_dir, epoch),
                        tag='{}'.format(epoch))
                    os.system("rm -rf {}/{}".format(model_dir, epoch))
        elif rank == 0:
            with open("{}/{}.yaml".format(model_dir, epoch), 'w') as fout:
                data = yaml.dump(infos)
                fout.write(data)
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(model, save_model_path, infos)

        final_epoch = epoch

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        try:
            if os.path.exists(final_model_path):
                os.remove(final_model_path)
            os.symlink('{}.pt'.format(final_epoch), final_model_path)
        except Exception:
            # On Windows, os.symlink might require admin; fall back to copy
            src = os.path.join(model_dir, f'{final_epoch}.pt')
            if os.path.exists(src):
                import shutil
                shutil.copyfile(src, final_model_path)
        writer.close()


if __name__ == '__main__':
    main()
