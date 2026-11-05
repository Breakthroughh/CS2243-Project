import torch

from src import checkpoint


def test_atomic_roundtrip(tmp_path):
    path = tmp_path / "latest.pt"
    checkpoint.save(path, model=torch.nn.Linear(2, 2).state_dict(), step=3)
    checkpoint.save(path, model=torch.nn.Linear(2, 2).state_dict(), step=4)
    assert checkpoint.load(path)["step"] == 4
    assert [p.name for p in tmp_path.iterdir()] == ["latest.pt"]


def test_save_runs_after_save_hooks(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(checkpoint, "on_save", [lambda: calls.append("commit")])
    checkpoint.save(tmp_path / "latest.pt", step=1)
    assert calls == ["commit"]


def test_rng_state_roundtrip():
    import random
    import numpy as np
    state = checkpoint.rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    checkpoint.set_rng_state(state)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == a
