"""Synthetic CPU integration: paired inputs, checkpoint loading, source tables and rendering."""
import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset

import analysis
from test_analysis import ToyPolicy


class Task(Dataset):
    path = Path("synthetic/task.hdf5")
    index = [("demo_1", t, 8) for t in range(8)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return dict(action=torch.full((3, 7), i / 8), state=torch.ones(2, 8), lang="synthetic")


class CheckpointPolicy(ToyPolicy):
    def __init__(self, loops):
        super().__init__(loops)
        self.register_buffer("alpha_bar_K", torch.linspace(0.99, 0.01, 100))

    def encode(self, batch):
        return None


def test_cpu_checkpoint_analysis_keeps_inputs_paired_and_renders(tmp_path, monkeypatch):
    import hydra.utils
    import src.data
    monkeypatch.setattr(src.data, "build_datasets", lambda *args: (None, ConcatDataset([Task()])))
    monkeypatch.setattr(hydra.utils, "instantiate", lambda cfg, **kw: CheckpointPolicy(cfg.n_loops))
    specs = []
    for loops in (2, 4, 6):
        path = tmp_path / f"checkpoint-{loops}"
        path.mkdir()
        policy_cfg = dict(n_loops=loops, n_train_steps=100)
        cfg = dict(policy=policy_cfg, data=dict(root="synthetic", download_dir="unused", action_horizon=3))
        (path / "config.yaml").write_text(json.dumps(cfg))
        torch.save(dict(policy=CheckpointPolicy(loops).state_dict(), policy_config=policy_cfg,
                        epoch=12, step=34), path / "latest.pt")
        specs.append(dict(run=f"synthetic-{loops}", loops=loops, path=str(path)))
    inputs = {r: [] for r in (2, 4, 6)}
    original = analysis.capture_loops
    def record(policy, noisy, *args):
        inputs[policy.n_loops].append(noisy.clone())
        return original(policy, noisy, *args)
    monkeypatch.setattr(analysis, "capture_loops", record)
    output = tmp_path / "output"
    analysis.analyze(specs, output, timesteps=(0, 50, 99), samples_per_task=4, batch_size=3, device="cpu")
    for loops in (4, 6):
        for reference, actual in zip(inputs[2], inputs[loops]):
            torch.testing.assert_close(reference, actual)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["samples"]) == 4
    assert [(c["epoch"], c["step"]) for c in manifest["checkpoints"]] == [(12, 34)] * 3
    with (output / "per_sample.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == (3 + 5 + 7) * 4 * 3
    final = [r for r in rows if r["loop"] == r["n_loops"]]
    assert all(abs(float(r["cosine"]) - 1) < 1e-6 for r in final)
    assert all(float(r["relative_l2"]) == 0 for r in final)
    for metric in ("cosine_similarity", "relative_l2"):
        for extension in ("png", "pdf"):
            assert (output / f"{metric}.{extension}").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        analysis.analyze(specs, output, device="cpu")
