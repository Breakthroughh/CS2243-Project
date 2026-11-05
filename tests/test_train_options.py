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


def test_resume_past_max_steps_does_not_train(cfg, libero_root, wandb_mock):
    # relaunching a finished run (same name -> resume from latest.pt) must not train past max_steps
    from src.logger import Logger
    Path(cfg.run_dir).mkdir(parents=True)
    cfg.train.max_steps = 3
    assert run(cfg, libero_root, Logger(cfg), epochs=1) == 3
    wandb_mock.log.reset_mock()
    assert run(cfg, libero_root, Logger(cfg), epochs=5) == 3
    assert not any("train/loss" in m for m, _ in logged(wandb_mock))


def test_grad_accumulation_equals_full_batch_update(cfg):
    from src.training.baseline import update
    from tests.test_train import Linear
    cfg.train.grad_clip = 1e9
    torch.manual_seed(0)
    full, accum = Linear(), Linear()
    accum.load_state_dict(full.state_dict())
    batch = {"state": torch.randn(4, 1, 8), "action": torch.randn(4, 1, 7)}
    halves = [{k: v[:2] for k, v in batch.items()}, {k: v[2:] for k, v in batch.items()}]
    out_full = update(cfg, full, torch.optim.SGD(full.parameters(), lr=0.1), [batch])
    out_accum = update(cfg, accum, torch.optim.SGD(accum.parameters(), lr=0.1), halves)
    assert torch.allclose(out_full["loss"], out_accum["loss"]) and torch.allclose(out_full["grad_norm"], out_accum["grad_norm"])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(full.parameters(), accum.parameters()))


def test_micro_batches_keep_steps_per_epoch(cfg, libero_root, wandb_mock, tmp_path):
    from src.logger import Logger
    assert cfg.train.micro_batch_size is None  # default: no accumulation
    Path(cfg.run_dir).mkdir(parents=True)
    full = run(cfg, libero_root, Logger(cfg), epochs=1)
    cfg.run_dir, cfg.train.micro_batch_size = str(tmp_path / "micro"), cfg.train.batch_size // 2
    Path(cfg.run_dir).mkdir(parents=True)
    assert run(cfg, libero_root, Logger(cfg), epochs=1) == full


class Noisy(Policy):
    """Loss depends on data order and on the torch RNG, so exact resume needs both restored."""

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 7)

    def compute_loss(self, batch):
        state_BS = batch["state"][:, -1] + 0.1 * torch.randn_like(batch["state"][:, -1])
        return {"loss": nn.functional.mse_loss(self.net(state_BS), batch["action"][:, 0])}


def test_resumed_run_matches_uninterrupted(cfg, libero_root, wandb_mock, tmp_path):
    from src.data import build_datasets
    from src.logger import Logger
    from src.training.baseline import train
    train_set, test_set = build_datasets(cfg, libero_root)

    def go(run_dir, epochs):
        cfg.run_dir, cfg.train.epochs = str(run_dir), epochs
        Path(run_dir).mkdir(exist_ok=True)
        torch.manual_seed(0)
        policy = Noisy()
        train(cfg, policy, train_set, test_set, Logger(cfg), evaluate=lambda p: {})
        return policy

    full = go(tmp_path / "full", 3)
    losses_full = [m["train/loss"] for m, _ in logged(wandb_mock) if "train/loss" in m]
    wandb_mock.log.reset_mock()
    go(tmp_path / "resumed", 1)  # interrupted after epoch 1
    torch.manual_seed(123)       # a resumed process starts from a different RNG state
    resumed = go(tmp_path / "resumed", 3)
    losses_resumed = [m["train/loss"] for m, _ in logged(wandb_mock) if "train/loss" in m]
    assert losses_resumed == losses_full
    assert all(torch.equal(a, b) for a, b in zip(full.state_dict().values(), resumed.state_dict().values()))
