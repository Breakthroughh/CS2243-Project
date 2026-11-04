"""Dims: B batch, O obs window, P action horizon, A action (7), V views, H height, W width, C rgb, L lang tokens (16),
E t5 width (768), N tokens."""
import pytest
import torch
from hydra.utils import instantiate

from src.config import compose
from src.policies.octo import OctoPolicy, attention_mask


@pytest.fixture(autouse=True)
def fake_t5(monkeypatch):
    monkeypatch.setattr("src.policies.octo.encode_language", lambda texts, device: torch.zeros(len(texts), 16, 768,
                                                                                              device=device))


def batch(B=2, O=1, P=4):
    return {"state": torch.zeros(B, O, 8), "image": torch.randint(0, 255, (B, O, 2, 128, 128, 3), dtype=torch.uint8),
            "lang": ["pick up the bowl"] * B, "action": torch.rand(B, P, 7) * 2 - 1}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("size, target", [("small", 27e6), ("base", 93e6)])
def test_param_counts_match_paper(size, target):
    assert abs(n_params(OctoPolicy(size=size)) - target) / target < 0.05


def test_attention_mask_rules():
    # prefix: 2 task tokens; per timestep: 3 obs tokens (2 image, 1 repeated lang), 1 readout; 2 timesteps
    allow_NN = attention_mask(n_task=2, n_obs=3, n_readout=1, window=2)
    task, obs0, read0, obs1, read1 = [0, 1], [2, 3, 4], [5], [6, 7, 8], [9]
    assert allow_NN[task][:, task].all() and not allow_NN[task][:, obs0 + read0].any()
    assert allow_NN[obs0][:, task + obs0].all() and not allow_NN[obs0][:, obs1 + read0].any()
    assert allow_NN[obs1][:, task + obs0 + obs1].all()
    assert allow_NN[read1][:, task + obs0 + obs1 + read0 + read1].all()
    assert not allow_NN[:, read0 + read1][[i for i in range(10) if i not in read0 + read1]].any()


def test_loss_backward_and_predict():
    policy = OctoPolicy(size="small")
    policy.set_action_stats(torch.zeros(7), torch.ones(7))
    out = policy.compute_loss(batch())
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert policy.head.reverse.out.weight.grad.abs().sum() > 0
    assert policy.stems["primary"].convs[0].weight.grad.abs().sum() > 0
    policy.eval()
    action_BPA = policy.predict_action(batch())
    assert action_BPA.shape == (2, 4, 7) and torch.isfinite(action_BPA).all()


def test_gripper_roundtrip_and_normalization():
    policy = OctoPolicy(size="small")
    policy.set_action_stats(torch.full((7,), 0.5), torch.full((7,), 2.0))
    action_BPA = torch.tensor([[[0.1, -0.2, 0.3, 0, 0, 0.5, -1.0], [0, 0, 0, 0, 0, 0, 1.0]]])
    model_BPA = policy.to_model(action_BPA)
    assert torch.allclose(model_BPA[..., :6], (action_BPA[..., :6] - 0.5) / 2.0)
    assert model_BPA[0, 0, 6] == 1 and model_BPA[0, 1, 6] == 0  # octo: 1 = open; libero: -1 = open
    assert torch.allclose(policy.from_model(model_BPA), action_BPA)


def test_hydra_configs():
    for size in ("small", "base"):
        cfg = compose([f"policy=octo_{size}"])
        assert cfg.data.obs_horizon == 1 and cfg.data.action_horizon == 4
        assert isinstance(instantiate(cfg.policy), OctoPolicy)


def test_pretrained_configs_load_converted_weights(tmp_path):
    ref = OctoPolicy(size="small")
    ref.save_pretrained(tmp_path)
    cfg = compose(["policy=octo_small_pt", f"policy.pretrained_model_name_or_path={tmp_path}"])
    assert cfg.data.obs_horizon == 1 and cfg.data.action_horizon == 4
    policy = instantiate(cfg.policy)
    assert isinstance(policy, OctoPolicy)
    assert all(torch.equal(a, b) for a, b in zip(policy.state_dict().values(), ref.state_dict().values()))
    assert compose(["policy=octo_base_pt"]).policy.pretrained_model_name_or_path == "jackfanml/octo-base-1.5-torch"
