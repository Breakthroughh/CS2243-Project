"""Baseline training loop. YOURS: make_optimizer, update (pure torch; defaults below are a plain BC step).
The rest is scaffolding: protocol logging, test loss, eval schedule, atomic checkpoint + resume (tests/test_train.py)."""
from pathlib import Path
import math

import torch
from torch.utils.data import DataLoader

from src import checkpoint


def lr_factor(t, step):
    """LR multiplier at optimizer step `step` (0-indexed): linear warmup from 0, then constant or cosine to 0 at
    max_steps (optax warmup_cosine_decay_schedule)."""
    if step < t.warmup_steps:
        return step / t.warmup_steps
    if t.schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * min((step - t.warmup_steps) / (t.max_steps - t.warmup_steps), 1)))
    return 1.0


def make_optimizer(cfg, policy):
    """AdamW, weight decay on weight matrices only (flax: kernels; not biases, norms, embeddings) -> (optimizer, scheduler)."""
    decay = lambda n, p: p.ndim >= 2 and n.endswith(("weight", "kernel"))
    params = list(policy.parameters())
    groups = [{"params": [p for n, p in policy.named_parameters() if decay(n, p)], "weight_decay": cfg.train.weight_decay},
              {"params": [p for n, p in policy.named_parameters() if not decay(n, p)], "weight_decay": 0.0}]
    assert sum(len(g["params"]) for g in groups) == len(params)
    optimizer = torch.optim.AdamW(groups, lr=cfg.train.lr)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(cfg.train, step))


def update(cfg, policy, optimizer, batches):
    """One optimizer step over equal-size micro-batches (gradient accumulation; 1 = plain step)
    -> {"loss", "grad_norm", **policy metrics} (scalars, averaged over micro-batches)."""
    optimizer.zero_grad()
    outs = []
    for batch in batches:
        out = policy.compute_loss(batch)
        (out["loss"] / len(batches)).backward()
        outs.append({k: v.detach() for k, v in out.items()})
    out = {k: torch.stack([o[k] for o in outs]).mean() for k in outs[0]}
    out["grad_norm"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip)
    optimizer.step()
    return out


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train(cfg, policy, train_set, test_set, logger, evaluate):
    """-> final global step. Resumes from run_dir/latest.pt if present."""
    path = Path(cfg.run_dir) / "latest.pt"
    optimizer, scheduler = make_optimizer(cfg, policy)
    start_epoch, step = 0, 0
    if path.exists():
        state = checkpoint.load(path)
        policy.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, step = state["epoch"], state["step"]
        if "rng" in state:  # absent in checkpoints written before RNG state was saved
            checkpoint.set_rng_state(state["rng"])

    micro = cfg.train.micro_batch_size or cfg.train.batch_size
    assert cfg.train.batch_size % micro == 0, "batch_size must be a multiple of micro_batch_size"
    accum = cfg.train.batch_size // micro
    order = torch.Generator()  # data order seeded per epoch: a resumed epoch replays exactly the same batches
    loader = DataLoader(train_set, micro, shuffle=True, drop_last=True, num_workers=cfg.train.num_workers, generator=order)
    test_loader = DataLoader(test_set, micro, num_workers=cfg.train.num_workers)

    done = lambda: cfg.train.max_steps is not None and step >= cfg.train.max_steps  # >=: resumed runs may already be past it
    for epoch in range(start_epoch, cfg.train.epochs):
        if done():
            break
        order.manual_seed(cfg.seed + epoch)
        policy.train()
        batches = []
        for batch in loader:
            batches.append(to_device(batch, cfg.device))
            if len(batches) < accum:
                continue
            lr = optimizer.param_groups[0]["lr"]
            out = update(cfg, policy, optimizer, batches)
            batches = []
            scheduler.step()
            step += 1
            logger.log({**{f"train/{k}": v.item() for k, v in out.items()}, "train/lr": lr,
                        "train/epoch": epoch, "train/step": step}, step)
            if done():
                break
                
        policy.eval()
        with torch.no_grad():
            outs = [policy.compute_loss(to_device(b, cfg.device)) for b in test_loader]
        logger.log({f"test/{k}": torch.stack([o[k] for o in outs]).mean().item() for k in outs[0]}, step)
        if (epoch + 1) % cfg.train.eval_every == 0:
            logger.log(evaluate(policy), step)
        checkpoint.save(path, policy=policy.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        epoch=epoch + 1, step=step, rng=checkpoint.rng_state())
        if done():
            break
    return step