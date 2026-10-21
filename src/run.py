"""One run: data, policy, logger -> train -> final eval -> optional HF push."""
from functools import partial
from pathlib import Path
import random
import re

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.config import compose  # noqa: F401
from src.data import build_datasets, resolve_root
from src.logger import Logger
from src.rollout import evaluate
from src.train import train


def main(cfg):
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    root = resolve_root(cfg)
    train_set, test_set = build_datasets(cfg, root)
    policy = instantiate(cfg.policy).to(cfg.device)
    logger = Logger(cfg)
    evaluate_fn = partial(evaluate, cfg=cfg, root=root)
    step = train(cfg, policy, train_set, test_set, logger, evaluate_fn)
    logger.log(evaluate_fn(policy), step)
    if cfg.hf.namespace:
        repo = re.sub(r"[^\w.-]", "-", f"{cfg.git.branch}-{cfg.name}")
        policy.push_to_hub(f"{cfg.hf.namespace}/{repo}", private=True, commit_message=f"{cfg.git.commit} {cfg.run_dir}")
    logger.finish()
