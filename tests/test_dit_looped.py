"""Looped DiT contract. Dims: B batch, P action horizon, A action, L context tokens, D width."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import nn

from src.config import compose
from src.policies.dit import DiTPolicy
from src.policies.dit_looped import LoopedDiTPolicy
from src.training.baseline import update


TINY = dict(action_horizon=4, obs_horizon=2, d_model=16, n_heads=2,
            n_train_steps=10, n_inference_steps=5, pretrained_vision=False, lang_model="mock")
SPLIT = dict(n_pre_layers=1, n_loop_layers=2, n_post_layers=1)


@pytest.fixture(autouse=True)
def mock_encoders(monkeypatch):
    """No downloads or pretrained models; exercise the real observation projection path."""
    class Tokens(dict):
        def to(self, device):
            return {k: v.to(device) for k, v in self.items()}

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embedding = nn.Embedding(2, 8)

        def forward(self, input_ids):
            return SimpleNamespace(pooler_output=self.embedding(input_ids).mean(1))

    monkeypatch.setattr("src.policies.dit.torchvision.models.resnet18",
                        lambda **kw: nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512)))
    monkeypatch.setattr("src.policies.dit.CLIPTokenizer.from_pretrained",
                        lambda *a: lambda texts, **kw: Tokens(input_ids=torch.zeros(len(texts), 2, dtype=torch.long)))
    monkeypatch.setattr("src.policies.dit.CLIPTextModel.from_pretrained", lambda *a: Language())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def batch():
    return dict(state=torch.randn(2, 2, 8), image=torch.zeros(2, 2, 2, 8, 8, 3, dtype=torch.uint8),
                lang=["pick", "place"], action=torch.rand(2, 4, 7) * 2 - 1)


def activate(policy):
    # Zero-initialized adaLN/output would hide recurrence bugs and block gradients at initialization.
    with torch.no_grad():
        for block in policy.blocks:
            nn.init.normal_(block.ada[1].weight, std=0.05)
        nn.init.normal_(policy.action_out.weight, std=0.05)


def test_one_loop_matches_baseline():
    baseline = DiTPolicy(**TINY, n_layers=4).eval()
    activate(baseline)
    looped = LoopedDiTPolicy(**TINY, **SPLIT, n_loops=1).eval()
    looped.load_state_dict(baseline.state_dict(), strict=True)
    x_BPA, k_B, ctx_BLD = torch.randn(2, 4, 7), torch.tensor([2, 7]), torch.randn(2, 5, 16)
    torch.testing.assert_close(looped.denoise(x_BPA, k_B, ctx_BLD), baseline.denoise(x_BPA, k_B, ctx_BLD))


def test_shared_recurrence_order_and_gradients():
    policy = LoopedDiTPolicy(**TINY, **SPLIT, n_loops=3)
    activate(policy)
    calls = []

    def record(index):
        def hook(module, inputs, output):
            output.retain_grad()
            calls.append((index, inputs, output))
        return hook

    handles = [block.register_forward_hook(record(i)) for i, block in enumerate(policy.blocks)]
    head = Mock(return_value=None)
    handles.append(policy.action_out.register_forward_hook(head))
    x_BPA, k_B, ctx_BLD = torch.randn(2, 4, 7), torch.tensor([2, 7]), torch.randn(2, 5, 16)
    policy.denoise(x_BPA, k_B, ctx_BLD).square().mean().backward()
    assert [i for i, _, _ in calls] == [0, 1, 2, 1, 2, 1, 2, 3]
    assert head.call_count == 1
    for previous, current in zip(calls, calls[1:]):
        assert current[1][0] is previous[2]
    for _, inputs, output in calls:
        assert inputs[1] is ctx_BLD
        assert inputs[2] is calls[0][1][2]
        assert output.grad is not None and output.grad.abs().sum() > 0
    assert policy.blocks[1].attn.in_proj_weight.grad.abs().sum() > 0
    for handle in handles:
        handle.remove()


def test_loop_count_does_not_add_parameters():
    models = [LoopedDiTPolicy(**TINY, **SPLIT, n_loops=r) for r in (1, 2, 4)]
    baseline = DiTPolicy(**TINY, n_layers=4)
    assert {sum(p.numel() for p in m.parameters()) for m in [baseline, *models]} == {
        sum(p.numel() for p in baseline.parameters())}
    assert all(list(m.state_dict()) == list(baseline.state_dict()) for m in models)


def test_loss_uses_only_final_noise_prediction(monkeypatch):
    policy = LoopedDiTPolicy(**TINY, **SPLIT, n_loops=2)
    activate(policy)
    b = batch()
    torch.manual_seed(5)
    k_B = torch.randint(0, 10, (2,))
    noise_BPA = torch.randn_like(b["action"])
    a_B11 = policy.alpha_bar_K[k_B][:, None, None]
    noisy_BPA = a_B11.sqrt() * b["action"] + (1 - a_B11).sqrt() * noise_BPA
    expected = F.mse_loss(policy.denoise(noisy_BPA, k_B, policy.encode(b)), noise_BPA)
    denoise = Mock(wraps=policy.denoise)
    monkeypatch.setattr(policy, "denoise", denoise)
    torch.manual_seed(5)
    out = policy.compute_loss(b)
    assert set(out) == {"loss"}
    assert denoise.call_count == 1
    torch.testing.assert_close(out["loss"], expected)


def test_config_defaults_and_training_update():
    default = compose(["policy=dit_looped"])
    assert (default.policy.n_pre_layers, default.policy.n_loop_layers, default.policy.n_post_layers,
            default.policy.n_loops) == (2, 2, 2, 2)
    cfg = compose(["policy=dit_looped", "device=cpu", "data.action_horizon=4", "policy.d_model=16",
                   "policy.n_heads=2", "policy.n_train_steps=10", "policy.n_inference_steps=5",
                   "policy.pretrained_vision=false", "policy.lang_model=mock"])
    policy = instantiate(cfg.policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr)
    before = policy.action_out.weight.detach().clone()
    out = update(cfg, policy, optimizer, batch())
    assert set(out) == {"loss", "grad_norm"}
    assert all(v.ndim == 0 and torch.isfinite(v) for v in out.values())
    assert not torch.equal(policy.action_out.weight, before)
    with torch.no_grad():
        action_BPA = policy.eval().predict_action(batch())
    assert action_BPA.shape == (2, 4, 7)
    assert torch.isfinite(action_BPA).all() and action_BPA.abs().max() <= 1


def test_hub_roundtrip(tmp_path):
    policy = LoopedDiTPolicy(**TINY, **SPLIT, n_loops=3).eval()
    activate(policy)
    policy.save_pretrained(tmp_path)
    loaded = LoopedDiTPolicy.from_pretrained(tmp_path).eval()
    assert (loaded.n_pre_layers, loaded.n_loop_layers, loaded.n_post_layers, loaded.n_loops) == (1, 2, 1, 3)
    b = batch()
    with torch.no_grad():
        torch.manual_seed(0)
        expected_BPA = policy.predict_action(b)
        torch.manual_seed(0)
        torch.testing.assert_close(loaded.predict_action(b), expected_BPA)


@pytest.mark.parametrize("kwargs", [{"n_loops": 0}, {"n_loops": 1.5}, {"n_pre_layers": -1},
                                   {"n_loop_layers": 0}, {"n_post_layers": -1}])
def test_invalid_loop_configuration(kwargs):
    with pytest.raises(ValueError):
        LoopedDiTPolicy(**TINY, **kwargs)


def test_empty_outer_stages():
    policy = LoopedDiTPolicy(**TINY, n_pre_layers=0, n_loop_layers=1, n_post_layers=0, n_loops=2)
    assert policy.denoise(torch.randn(2, 4, 7), torch.tensor([1, 2]), torch.randn(2, 5, 16)).shape == (2, 4, 7)
