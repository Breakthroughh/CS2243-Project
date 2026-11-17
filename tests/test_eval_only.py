"""Eval-only loop contract. Dims: R refinement loops, B batch, P action horizon, A action, L context, D width."""
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from src import checkpoint
from src.config import compose
from src.policies.dit_looped_xsa import LoopedDiTXSAPolicy
from src.run import main
from src.training.eval_only import train
from tests.test_dit_looped import SPLIT, TINY, activate, mock_encoders  # noqa: F401
from tests.conftest import logged
from tests.test_train import Linear


def test_config_selects_loop_and_requires_checkpoint():
    cfg = compose(["train=eval_only", "train.checkpoint=/results/runs/exp/x/y/latest.pt"])
    assert cfg.train.loop == "src.training.eval_only.train"
    with pytest.raises(Exception):
        compose(["train=eval_only"]).train.checkpoint  # mandatory: no silent default


def test_loads_weights_and_step_without_training_or_writing(cfg, tmp_path):
    source = Linear()
    path = tmp_path / "other_run" / "latest.pt"
    path.parent.mkdir()
    checkpoint.save(path, policy=source.state_dict(), epoch=3, step=123)
    saved = path.read_bytes()
    cfg.train = {"loop": "src.training.eval_only.train", "checkpoint": str(path)}
    Path(cfg.run_dir).mkdir(parents=True)
    policy, logger, evaluate = Linear(), Mock(), Mock()

    assert train(cfg, policy, None, None, logger, evaluate) == 123
    for name, weight in source.state_dict().items():
        torch.testing.assert_close(policy.state_dict()[name], weight)
    logger.log.assert_not_called()  # no train/* or test/loss rows
    evaluate.assert_not_called()  # src/run.py runs the single final eval
    assert not (Path(cfg.run_dir) / "latest.pt").exists() and path.read_bytes() == saved


@pytest.mark.parametrize("n_loops", [1, 3, 6])
def test_fewer_inference_loops_reproduce_the_trained_intermediate_exit(cfg, tmp_path, n_loops):
    """A 6-loop checkpoint run with n_loops=r decodes exactly the loop-r prediction deep supervision trained."""
    trained = LoopedDiTXSAPolicy(**TINY, **SPLIT, n_loops=6).eval()
    activate(trained)
    path = tmp_path / "latest.pt"
    checkpoint.save(path, policy=trained.state_dict(), epoch=1, step=7)
    cfg.train = {"loop": "src.training.eval_only.train", "checkpoint": str(path)}
    policy = LoopedDiTXSAPolicy(**TINY, **SPLIT, n_loops=n_loops).eval()
    assert train(cfg, policy, None, None, Mock(), Mock()) == 7

    x_BPA, k_B, ctx_BLD = torch.randn(2, 4, 7), torch.tensor([2, 7]), torch.randn(2, 5, 16)
    predictions_RBPA = trained.denoise_loops(x_BPA, k_B, ctx_BLD)
    torch.testing.assert_close(policy.denoise(x_BPA, k_B, ctx_BLD), predictions_RBPA[n_loops - 1])


def test_run_evaluates_checkpoint_once_at_its_step(cfg, tmp_path, fake_tasks, wandb_mock):
    """Through src/run.py: a new run dir, another run's weights, a different loop count -> one eval row."""
    kwargs = dict(TINY, **SPLIT)
    del kwargs["action_horizon"], kwargs["obs_horizon"]
    trained = LoopedDiTXSAPolicy(**TINY, **SPLIT, n_loops=6)
    path = tmp_path / "latest.pt"
    checkpoint.save(path, policy=trained.state_dict(), epoch=1, step=7)
    cfg.train = {"loop": "src.training.eval_only.train", "checkpoint": str(path)}
    cfg.policy = {"_target_": "src.policies.dit_looped_xsa.LoopedDiTXSAPolicy", "action_horizon": 4,
                  "obs_horizon": 2, "n_loops": 2, **kwargs}
    main(cfg)
    rows = logged(wandb_mock)
    assert [step for _, step in rows] == [7]
    assert "test/mean_score" in rows[0][0] and not any(k.startswith("train/loss") for k in rows[0][0])
    assert not (Path(cfg.run_dir) / "latest.pt").exists()
