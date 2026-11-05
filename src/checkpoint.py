"""Atomic torch checkpoints (write tmp, then rename) + RNG state for exact resume.
on_save: callables run after every save (compute/app.py registers volume.commit, so checkpoints survive a hard kill)."""
import os
import random

import numpy as np
import torch

on_save = []


def save(path, **state):
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    for hook in on_save:
        hook()


def load(path):
    return torch.load(path, weights_only=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
