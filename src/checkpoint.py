"""Atomic torch checkpoints: write tmp, then rename."""
import os

import torch
from omegaconf import OmegaConf


def validate_policy(cfg, saved_policy):
    if saved_policy != OmegaConf.to_container(cfg.policy, resolve=True):
        raise ValueError("Policy configuration differs or is unavailable; use a new run name.")


def save(path, **state):
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path):
    return torch.load(path, weights_only=False)
