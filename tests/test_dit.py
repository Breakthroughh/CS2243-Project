"""Contract for src/dit.py. Dims: B batch, O obs horizon, P action horizon, V views, H height, W width, C rgb,
S state (8), A action (7)."""
import torch
from hydra.utils import instantiate

from src.dit import DiTPolicy
from src.policy import Policy
from src.run import compose

TINY = dict(action_horizon=4, obs_horizon=2, d_model=32, n_layers=2, n_heads=2, n_train_steps=10,
            n_inference_steps=5, pretrained_vision=False, lang_model="hf-internal-testing/tiny-random-CLIPModel")


def batch(B=3, P=4):
    g = torch.Generator().manual_seed(0)
    return {"state": torch.randn(B, 2, 8, generator=g),  # [B, O, S]
            "image": torch.randint(0, 256, (B, 2, 2, 8, 8, 3), dtype=torch.uint8, generator=g),  # [B, O, V, H, W, C]
            "lang": ["pick up the bowl", "open the drawer", "pick up the bowl"][:B],
            "action": torch.rand(B, P, 7, generator=g) * 2 - 1}  # [B, P, A] in [-1, 1]


def test_from_config():
    cfg = compose(["policy=dit", "device=cpu", "data.action_horizon=4", "policy.d_model=32", "policy.n_layers=2",
                   "policy.n_heads=2", "policy.pretrained_vision=false", f"policy.lang_model={TINY['lang_model']}"])
    policy = instantiate(cfg.policy)
    assert isinstance(policy, Policy)
    assert policy.predict_action(batch()).shape == (3, 4, 7)


def test_predict_shape_and_range():
    policy = DiTPolicy(**TINY).eval()
    action_BPA = policy.predict_action(batch())
    assert action_BPA.shape == (3, 4, 7)
    assert action_BPA.abs().max() <= 1


def test_loss_is_finite_scalar():
    loss = DiTPolicy(**TINY).compute_loss(batch())["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    policy, b = DiTPolicy(**TINY), batch()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    losses = []
    for _ in range(200):
        loss = policy.compute_loss(b)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert sum(losses[-20:]) < 0.5 * sum(losses[:20])


def test_hub_roundtrip(tmp_path):
    policy = DiTPolicy(**TINY).eval()
    policy.save_pretrained(tmp_path)
    loaded = DiTPolicy.from_pretrained(tmp_path).eval()
    torch.manual_seed(0)
    expected_BPA = policy.predict_action(batch())
    torch.manual_seed(0)
    assert torch.allclose(loaded.predict_action(batch()), expected_BPA)
