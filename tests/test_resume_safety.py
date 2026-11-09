"""Resume limits and experiment identity. Dims: B batch, P action horizon, A action."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from src import checkpoint
from src.data import build_datasets
from src.run import main
from src.training.baseline import train
from tests.test_train import Linear


def setup_training(cfg, libero_root):
    cfg.train.epochs = 3
    Path(cfg.run_dir).mkdir(parents=True)
    return build_datasets(cfg, libero_root)


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_fresh_run_obeys_step_limit(cfg, libero_root, limit):
    train_set, test_set = setup_training(cfg, libero_root)
    cfg.train.max_steps = limit
    logger = Mock()
    assert train(cfg, Linear(), train_set, test_set, logger, evaluate=lambda p: {}) == limit
    updates = [call for call in logger.log.call_args_list if "train/step" in call.args[0]]
    assert len(updates) == limit


@pytest.mark.parametrize("new_limit", [0, 1])
def test_completed_or_over_limit_checkpoint_does_not_update(cfg, libero_root, new_limit):
    train_set, test_set = setup_training(cfg, libero_root)
    cfg.train.max_steps = 1
    assert train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {}) == 1
    path = Path(cfg.run_dir) / "latest.pt"
    previous = path.read_bytes()
    cfg.train.max_steps = new_limit
    logger = Mock()
    assert train(cfg, Linear(), train_set, test_set, logger, evaluate=lambda p: {}) == 1
    logger.log.assert_not_called()
    assert path.read_bytes() == previous


def test_increasing_limit_allows_same_policy_to_resume(cfg, libero_root):
    train_set, test_set = setup_training(cfg, libero_root)
    cfg.train.max_steps = 1
    assert train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {}) == 1
    cfg.train.max_steps = 2
    assert train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {}) == 2


@pytest.mark.parametrize("change", ["target", "loop_count"])
def test_rejects_changed_policy_before_overwriting_config(cfg, monkeypatch, change):
    cfg.policy = OmegaConf.create({"_target_": "src.policies.dit_looped.LoopedDiTPolicy", "n_loops": 2})
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True)
    path = run_dir / "config.yaml"
    original = OmegaConf.to_yaml(cfg)
    path.write_text(original)
    checkpoint_path = run_dir / "latest.pt"
    checkpoint_path.write_bytes(b"checkpoint must remain untouched")
    if change == "target":
        cfg.policy._target_ = "src.policies.dit_looped_deepsup.DeepSupervisedLoopedDiTPolicy"
    else:
        cfg.policy.n_loops = 4
    resolve_root = Mock(side_effect=AssertionError("Data loading must not start"))
    monkeypatch.setattr("src.run.resolve_root", resolve_root)
    with pytest.raises(ValueError, match="[Pp]olicy.*new run name"):
        main(cfg)
    resolve_root.assert_not_called()
    assert path.read_text() == original
    assert checkpoint_path.read_bytes() == b"checkpoint must remain untouched"


def test_direct_training_rejects_changed_policy(cfg, libero_root):
    train_set, test_set = setup_training(cfg, libero_root)
    cfg.train.max_steps = 1
    assert train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {}) == 1
    path = Path(cfg.run_dir) / "latest.pt"
    previous = path.read_bytes()
    cfg.policy._target_ = "src.policies.dit_looped_deepsup.DeepSupervisedLoopedDiTPolicy"
    cfg.train.max_steps = 2
    with pytest.raises(ValueError, match="[Pp]olicy.*new run name"):
        train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {})
    assert path.read_bytes() == previous


def test_legacy_checkpoint_resumes_using_saved_config(cfg, libero_root):
    train_set, test_set = setup_training(cfg, libero_root)
    cfg.train.max_steps = 1
    assert train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {}) == 1
    path = Path(cfg.run_dir) / "latest.pt"
    state = checkpoint.load(path)
    state.pop("policy_config", None)
    checkpoint.save(path, **state)
    (path.parent / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    cfg.train.max_steps = 2
    assert train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {}) == 2


def test_legacy_checkpoint_without_config_is_rejected(cfg, libero_root):
    train_set, test_set = setup_training(cfg, libero_root)
    cfg.train.max_steps = 1
    assert train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {}) == 1
    path = Path(cfg.run_dir) / "latest.pt"
    state = checkpoint.load(path)
    state.pop("policy_config", None)
    checkpoint.save(path, **state)
    cfg.train.max_steps = 2
    with pytest.raises(ValueError, match="[Pp]olicy.*new run name"):
        train(cfg, Linear(), train_set, test_set, Mock(), evaluate=lambda p: {})
