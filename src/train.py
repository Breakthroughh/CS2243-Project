"""Training loop: placeholder for you to implement. Contract: tests/test_train.py (drop its xfail marker when done).

train(cfg, policy, train_set, test_set, logger, evaluate) -> final global step
- logger.log({"train/loss", "train/lr", "train/grad_norm", "train/epoch", "train/step", **train/<policy metric>}, step)
- each epoch: logger.log({"test/loss", ...}, step) on test_set (held-out demos)
- every cfg.train.eval_every epochs: logger.log(evaluate(policy), step)
- each epoch: src.checkpoint.save(Path(cfg.run_dir) / "latest.pt", ...); resume from it if it exists
"""


def train(cfg, policy, train_set, test_set, logger, evaluate):
    if cfg.train.epochs == 0:
        return 0
    raise NotImplementedError("implement src/train.py:train (contract: tests/test_train.py)")