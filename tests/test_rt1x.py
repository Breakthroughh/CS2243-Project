"""Dims: B batch, O obs window, A action (7), K action tokens (11), V views, H height, W width, C rgb, N tokens."""
import pytest
import torch

from src.config import compose
from src.policies.rt1x import RT1XPolicy, attention_mask, detokenize, tokenize


@pytest.fixture(autouse=True)
def fake_minilm(monkeypatch):
    # nonzero: FiLM is zero-initialised, so a zero context would give its weights zero gradient
    monkeypatch.setattr("src.policies.rt1x.encode_language", lambda texts, device: torch.ones(len(texts), 384,
                                                                                              device=device))


def batch(B=2, O=2):
    return {"state": torch.zeros(B, O, 8), "image": torch.randint(0, 255, (B, O, 2, 128, 128, 3), dtype=torch.uint8),
            "lang": ["pick up the bowl"] * B, "action": torch.rand(B, O, 7) * 2 - 1}


def test_param_count_matches_released_rt1x_config():
    n = sum(p.numel() for p in RT1XPolicy(obs_horizon=15, pretrained=False).parameters())
    assert 50e6 < n < 65e6


def test_attention_mask_only_attends_past_image_tokens():
    allow_NN = attention_mask(n_image=2, n_action=3, window=2)
    image = torch.tensor([1, 1, 0, 0, 0] * 2, dtype=torch.bool)
    causal_NN = torch.ones(10, 10, dtype=torch.bool).tril()
    assert torch.equal(allow_NN, causal_NN & image[None, :])


def test_tokenize_roundtrip():
    action_BA = torch.rand(64, 7) * 2 - 1
    tokens_BK = tokenize(action_BA, vocab=512)
    assert tokens_BK.shape == (64, 11) and tokens_BK.min() >= 0 and tokens_BK.max() < 512
    assert (tokens_BK[:, 0] == 1).all()  # mode: arm + gripper
    assert (detokenize(tokens_BK, vocab=512) - action_BA).abs().max() < 4 / 511


def test_loss_backward_and_predict():
    policy = RT1XPolicy(obs_horizon=2, pretrained=False)
    out = policy.compute_loss(batch())
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert policy.logits.weight.grad.abs().sum() > 0
    assert policy.tokenizer.film[0].mul.weight.grad.abs().sum() > 0
    policy.eval()
    action_BPA = policy.predict_action(batch())
    assert action_BPA.shape == (2, 1, 7) and (action_BPA.abs() <= 1).all()


def test_hydra_config():
    cfg = compose(["policy=rt1x"])
    assert cfg.data.obs_horizon == 15 and cfg.data.action_horizon == 15 and cfg.data.action_offset == 14
    assert cfg.eval.n_action_steps == 1 and cfg.policy.obs_horizon == 15


def test_hydra_config_1_frame():
    cfg = compose(["policy=rt1x_1f"])
    assert (cfg.data.obs_horizon, cfg.data.action_horizon, cfg.data.action_offset) == (1, 1, 0)
    assert cfg.eval.n_action_steps == 1 and cfg.policy.obs_horizon == 1
