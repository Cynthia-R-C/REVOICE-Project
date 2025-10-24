# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Tsinghua Univ. (authors: Xingchen Song)
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

# ==================== train_utils.py (DeepSpeed-free Windows patch FINAL COMPLETE) ====================
# Fully compatible with single-GPU training on Windows for interspeech2024-code.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def wenet_join(model, optimizer, scheduler, args):
    """Return model/optimizer/scheduler triple — DeepSpeed disabled."""
    print("[Warning] DeepSpeed not installed — running without distributed optimizer.")
    return model, optimizer, scheduler


def batch_forward(model, batch, criterion, accum_grad=1):
    """Compute forward loss."""
    feats, feats_lengths, label, label_lengths = batch
    loss, loss_att, loss_ctc, _ = model(
        feats, feats_lengths, label, label_lengths)
    loss = loss / accum_grad
    loss.backward()
    return loss, loss_att, loss_ctc


def batch_backward(optimizer, scheduler, step, accum_grad=1):
    """Perform backward + optimization."""
    if (step + 1) % accum_grad == 0:
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()


def update_parameter_and_lr(model, optimizer, scheduler, step, accum_grad=1, max_norm=None):
    """Simple PyTorch-based parameter update function (replacement for DeepSpeed logic)."""
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    if (step + 1) % accum_grad == 0:
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
    return scheduler.get_last_lr()[0] if scheduler else None


def compute_grad_norm(model):
    """Compute gradient norm for monitoring."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    return total_norm


def clip_grad_norm(model, max_norm):
    """Clip gradient norm if needed."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def sync_model(model):
    """Placeholder for distributed synchronization (unused)."""
    pass


def load_average_model(model, ckpt_list):
    """Average weights from multiple checkpoints."""
    print("[Info] Averaging checkpoints:", len(ckpt_list))
    state_dicts = [torch.load(c, map_location="cpu") for c in ckpt_list]
    avg_state = state_dicts[0]
    for k in avg_state:
        for s in state_dicts[1:]:
            avg_state[k] += s[k]
        avg_state[k] /= len(state_dicts)
    model.load_state_dict(avg_state)
    print("[Info] Loaded averaged model weights.")
    return model


def compute_loss(loss, loss_att, loss_ctc, weight=0.3):
    """Combine attention + CTC losses."""
    return weight * loss_ctc + (1 - weight) * loss_att


def save_training_state(model, optimizer, scheduler, epoch, path):
    """Save checkpoint."""
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, path)
    print(f"[Info] Saved training state → {path}")


def load_training_state(model, optimizer, scheduler, path):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[Info] Loaded training state ← {path}")
    return ckpt.get("epoch", 0)


def save_model(model, path):
    """Save model weights only (used by Executor)."""
    torch.save(model.state_dict(), path)
    print(f"[Info] Model weights saved to {path}")


def log_per_step(step, loss, grad_norm=None, lr=None):
    """Simple step-level logger used by Executor."""
    msg = f"[Step {step}] Loss={loss:.4f}"
    if grad_norm is not None:
        msg += f" | GradNorm={grad_norm:.4f}"
    if lr is not None:
        msg += f" | LR={lr:.6f}"
    print(msg)
