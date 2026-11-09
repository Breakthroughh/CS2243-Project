"""Deep supervision contract. Dims: R refinement loops, B batch, P action horizon, A action, L context, D width."""
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from src.config import compose
from src.logger import Logger
from src.policies.dit_looped import LoopedDiTPolicy
from src.policies.dit_looped_deepsup import DeepSupervisedLoopedDiTPolicy
from src.training.baseline import train, update
from tests.conftest import logged
from tests.test_dit_looped import SPLIT, TINY, activate, batch, mock_encoders


@pytest.mark.parametrize("n_loops", [1, 2, 4])
def test_predictions_match_final_only_arm_without_new_parameters(n_loops):
    reference = LoopedDiTPolicy(**TINY, **SPLIT, n_loops=n_loops).eval()
    activate(reference)
    policy = DeepSupervisedLoopedDiTPolicy(**TINY, **SPLIT, n_loops=n_loops).eval()
    policy.load_state_dict(reference.state_dict(), strict=True)
    assert sum(p.numel() for p in policy.parameters()) == sum(p.numel() for p in reference.parameters())
    x_BPA, k_B, ctx_BLD = torch.randn(2, 4, 7), torch.tensor([2, 7]), torch.randn(2, 5, 16)
    predictions_RBPA = policy.denoise_loops(x_BPA, k_B, ctx_BLD)
    assert predictions_RBPA.shape == (n_loops, 2, 4, 7)
    torch.testing.assert_close(predictions_RBPA[-1], reference.denoise(x_BPA, k_B, ctx_BLD))
    torch.testing.assert_close(policy.denoise(x_BPA, k_B, ctx_BLD), predictions_RBPA[-1])


def test_shared_decoder_is_a_branch_not_recurrent_input():
    policy = DeepSupervisedLoopedDiTPolicy(**TINY, **SPLIT, n_loops=3)
    activate(policy)
    calls = []

    def record(index):
        def hook(module, inputs, output):
            calls.append((index, inputs, output))
        return hook

    handles = [block.register_forward_hook(record(i)) for i, block in enumerate(policy.blocks)]
    head = Mock(return_value=None)
    handles.append(policy.action_out.register_forward_hook(head))
    x_BPA, k_B, ctx_BLD = torch.randn(2, 4, 7), torch.tensor([2, 7]), torch.randn(2, 5, 16)
    predictions_RBPA = policy.denoise_loops(x_BPA, k_B, ctx_BLD)
    assert [i for i, _, _ in calls] == [0, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert head.call_count == 3
    assert calls[4][1][0] is calls[2][2]
    assert calls[7][1][0] is calls[5][2]
    for _, inputs, _ in calls:
        assert inputs[1] is ctx_BLD
        assert inputs[2] is calls[0][1][2]
    # Each exit independently trains the shared refinement, post-loop blocks, and head.
    parameters = [policy.blocks[1].attn.in_proj_weight, policy.blocks[3].attn.in_proj_weight,
                  policy.action_out.weight]
    for prediction_BPA in predictions_RBPA:
        grads = torch.autograd.grad(prediction_BPA.square().mean(), parameters, retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("n_loops", [1, 3])
def test_mean_loss_shared_noise_target_and_per_loop_metrics(monkeypatch, n_loops):
    policy = DeepSupervisedLoopedDiTPolicy(**TINY, **SPLIT, n_loops=n_loops)
    activate(policy)
    b = batch()
    torch.manual_seed(5)
    k_B = torch.randint(0, 10, (2,))
    noise_BPA = torch.randn_like(b["action"])
    a_B11 = policy.alpha_bar_K[k_B][:, None, None]
    noisy_BPA = a_B11.sqrt() * b["action"] + (1 - a_B11).sqrt() * noise_BPA
    predictions_RBPA = policy.denoise_loops(noisy_BPA, k_B, policy.encode(b))
    expected_R = torch.stack([F.mse_loss(prediction_BPA, noise_BPA) for prediction_BPA in predictions_RBPA])
    encode = Mock(wraps=policy.encode)
    denoise_loops = Mock(wraps=policy.denoise_loops)
    monkeypatch.setattr(policy, "encode", encode)
    monkeypatch.setattr(policy, "denoise_loops", denoise_loops)
    torch.manual_seed(5)
    out = policy.compute_loss(b)
    assert encode.call_count == denoise_loops.call_count == 1
    torch.testing.assert_close(denoise_loops.call_args.args[0], noisy_BPA)
    torch.testing.assert_close(denoise_loops.call_args.args[1], k_B)
    assert set(out) == {"loss", *(f"loss_loop_{r + 1}" for r in range(n_loops))}
    torch.testing.assert_close(out["loss"], expected_R.mean())
    for r, expected in enumerate(expected_R, 1):
        torch.testing.assert_close(out[f"loss_loop_{r}"], expected)
        assert not out[f"loss_loop_{r}"].requires_grad
    out["loss"].backward()
    assert policy.blocks[1].attn.in_proj_weight.grad.abs().sum() > 0


def test_sampler_only_decodes_final_exit(monkeypatch):
    policy = DeepSupervisedLoopedDiTPolicy(**TINY, **SPLIT, n_loops=3).eval()
    activate(policy)
    reference = LoopedDiTPolicy(**TINY, **SPLIT, n_loops=3).eval()
    reference.load_state_dict(policy.state_dict())
    monkeypatch.setattr(policy, "denoise_loops", Mock(side_effect=AssertionError("training exits used in sampling")))
    head = Mock(return_value=None)
    handle = policy.action_out.register_forward_hook(head)
    b = batch()
    with torch.no_grad():
        torch.manual_seed(0)
        actual_BPA = policy.predict_action(b)
        torch.manual_seed(0)
        expected_BPA = reference.predict_action(b)
    torch.testing.assert_close(actual_BPA, expected_BPA)
    assert head.call_count == policy.n_inference_steps
    handle.remove()


def test_config_and_optimizer_update():
    cfg = compose(["policy=dit_looped_deepsup", "device=cpu", "data.action_horizon=4", "policy.d_model=16",
                   "policy.n_heads=2", "policy.n_train_steps=10", "policy.n_inference_steps=5",
                   "policy.pretrained_vision=false", "policy.lang_model=mock"])
    assert (cfg.policy.n_pre_layers, cfg.policy.n_loop_layers, cfg.policy.n_post_layers, cfg.policy.n_loops) == (2, 2, 2, 2)
    policy = instantiate(cfg.policy)
    before = policy.action_out.weight.detach().clone()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr)
    out = update(cfg, policy, optimizer, batch())
    assert all(v.ndim == 0 and torch.isfinite(v) for v in out.values())
    assert not torch.equal(policy.action_out.weight, before)


def test_hub_roundtrip_preserves_loop_config_and_losses(tmp_path):
    policy = DeepSupervisedLoopedDiTPolicy(**TINY, **SPLIT, n_loops=3).eval()
    activate(policy)
    policy.save_pretrained(tmp_path)
    loaded = DeepSupervisedLoopedDiTPolicy.from_pretrained(tmp_path).eval()
    assert (loaded.n_pre_layers, loaded.n_loop_layers, loaded.n_post_layers, loaded.n_loops) == (1, 2, 1, 3)
    b = batch()
    torch.manual_seed(0)
    expected = policy.compute_loss(b)
    torch.manual_seed(0)
    actual = loaded.compute_loss(b)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_training_logs_loop_losses_and_resumes(cfg, wandb_mock):
    cfg.train.epochs, cfg.train.num_workers, cfg.train.batch_size = 1, 0, 2
    Path(cfg.run_dir).mkdir(parents=True)
    b = batch()
    dataset = [{key: value[i] for key, value in b.items()} for i in range(2)]
    policy = DeepSupervisedLoopedDiTPolicy(**TINY, **SPLIT, n_loops=2)
    logger = Logger(cfg)
    assert train(cfg, policy, dataset, dataset, logger, evaluate=lambda p: {}) == 1
    metrics = set().union(*(m for m, _ in logged(wandb_mock)))
    assert {"train/loss", "train/loss_loop_1", "train/loss_loop_2",
            "test/loss", "test/loss_loop_1", "test/loss_loop_2"} <= metrics
    cfg.train.epochs = 2
    resumed = DeepSupervisedLoopedDiTPolicy(**TINY, **SPLIT, n_loops=2)
    assert train(cfg, resumed, dataset, dataset, logger, evaluate=lambda p: {}) == 2
