"""Eval-only loop: load another run's checkpoint and skip training; src/run.py then runs the one final eval.
Policy kwargs that leave the weights unchanged (n_loops, n_inference_steps) may differ from the checkpoint's run."""
from src import checkpoint


def train(cfg, policy, train_set, test_set, logger, evaluate):
    """-> the checkpoint's global step. Writes nothing to run_dir, so a relaunch just re-evaluates."""
    state = checkpoint.load(cfg.train.checkpoint)
    policy.load_state_dict(state["policy"])
    return state["step"]
