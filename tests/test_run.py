from pathlib import Path

from omegaconf import OmegaConf

from src.policy import NoOpPolicy
from src.run import main
from tests.conftest import logged


def test_noop_baseline_end_to_end(cfg, fake_tasks, wandb_mock, monkeypatch):
    pushes = []
    monkeypatch.setattr(NoOpPolicy, "push_to_hub", lambda self, repo, **k: pushes.append(repo))
    main(cfg)
    assert OmegaConf.load(Path(cfg.run_dir) / "config.yaml").name == "test"
    metrics, step = logged(wandb_mock)[-1]
    assert step == 0 and metrics["test/mean_score"] == 0.5
    assert pushes == []
    cfg.hf.namespace = "me"
    main(cfg)
    assert pushes == ["me/exp-test-test"]
