"""Train options on the baseline loop: weight-decay groups, warmup + constant/cosine LR, scheduler resume.
Dims: S state (8), A action (7)."""
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from src.config import compose
from src.policies.baseline import Policy
from src.training.baseline import lr_factor, make_optimizer
from tests.conftest import logged
from tests.test_train import run


class Mixed(Policy):
    def __init__(self):
        super().__init__()
        self.net, self.norm = nn.Linear(8, 7), nn.LayerNorm(7)
        self.pos = nn.Parameter(torch.zeros(1, 4, 7))


def test_weight_decay_only_on_weight_matrices(cfg):
    cfg.train.weight_decay = 0.1
    optimizer, _ = make_optimizer(cfg, Mixed())
    decay = {id(p) for g in optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    policy_params = dict(Mixed().named_parameters())
    assert len(decay) == 1 and sum(len(g["params"]) for g in optimizer.param_groups) == len(policy_params)
    assert all(g["weight_decay"] in (0.0, 0.1) for g in optimizer.param_groups)


@pytest.mark.parametrize("schedule, expected", [
    ("cosine", {0: 0.0, 5: 0.5, 10: 1.0, 60: 0.5, 110: 0.0}),
    ("constant", {0: 0.0, 5: 0.5, 10: 1.0, 60: 1.0, 110: 1.0}),
])
def test_lr_factor_warmup_then_schedule(cfg, schedule, expected):
    cfg.train.schedule, cfg.train.warmup_steps, cfg.train.max_steps = schedule, 10, 110
    for step, factor in expected.items():
        assert math.isclose(lr_factor(cfg.train, step), factor, abs_tol=1e-9)


def test_default_is_constant_lr(cfg):
    assert lr_factor(cfg.train, 0) == lr_factor(cfg.train, 10_000) == 1.0


def test_logged_lr_follows_schedule_across_resume(cfg, libero_root, wandb_mock):
    from src.logger import Logger
    Path(cfg.run_dir).mkdir(parents=True)
    cfg.train.schedule, cfg.train.warmup_steps, cfg.train.max_steps = "cosine", 3, 40
    run(cfg, libero_root, Logger(cfg), epochs=1)
    run(cfg, libero_root, Logger(cfg), epochs=2)
    lrs = {s: m["train/lr"] for m, s in logged(wandb_mock) if "train/lr" in m}
    assert len(lrs) > 3 and all(math.isclose(lr, cfg.train.lr * lr_factor(cfg.train, s - 1)) for s, lr in lrs.items())


def test_octo_train_config():
    t = compose(["train=octo"]).train
    assert (t.lr, t.warmup_steps, t.schedule, t.max_steps, t.batch_size, t.weight_decay, t.grad_clip) == \
        (3e-4, 2000, "cosine", 50_000, 256, 0.01, 1.0)
    assert t.loop == "src.training.baseline.train"
