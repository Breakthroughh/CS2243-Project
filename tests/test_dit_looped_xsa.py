"""XSA contract. Dims: B batch, J attention heads, P action tokens, E head width, D model width."""
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from torch import nn

from src.config import compose
from src.policies.dit_looped_deepsup import DeepSupervisedLoopedDiTPolicy
from src.policies.dit_looped_xsa import ExclusiveSelfAttention, LoopedDiTXSAPolicy, exclusive_projection
from src.training.baseline import update
from tests.test_dit_looped import SPLIT, TINY, activate, batch, mock_encoders


def test_projection_matches_geometry_and_does_not_increase_norm():
    torch.manual_seed(0)
    output_BJPE, value_BJPE = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 8)
    result_BJPE = exclusive_projection(output_BJPE, value_BJPE)
    torch.testing.assert_close((result_BJPE * value_BJPE).sum(-1), torch.zeros(2, 3, 4), atol=2e-6, rtol=0)
    assert (result_BJPE.norm(dim=-1) <= output_BJPE.norm(dim=-1) + 1e-6).all()
    torch.testing.assert_close(exclusive_projection(value_BJPE, value_BJPE), torch.zeros_like(value_BJPE),
                               atol=1e-6, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_zero_values_are_safe_and_preserve_output(dtype):
    output_BJPE = torch.randn(2, 2, 3, 4, dtype=dtype, requires_grad=True)
    value_BJPE = torch.zeros_like(output_BJPE, requires_grad=True)
    result_BJPE = exclusive_projection(output_BJPE, value_BJPE)
    assert result_BJPE.dtype == dtype
    torch.testing.assert_close(result_BJPE, output_BJPE)
    result_BJPE.float().square().mean().backward()
    assert torch.isfinite(output_BJPE.grad).all() and torch.isfinite(value_BJPE.grad).all()


def test_projection_gradients():
    torch.manual_seed(0)
    args = [torch.randn(1, 2, 2, 3, dtype=torch.double, requires_grad=True) for _ in range(2)]
    assert torch.autograd.gradcheck(exclusive_projection, args)


def test_attention_projects_each_head_before_output_projection():
    torch.manual_seed(0)
    attention = nn.MultiheadAttention(8, 2, batch_first=True)
    xsa = ExclusiveSelfAttention(attention)
    x_BPD = torch.randn(2, 3, 8, requires_grad=True)
    q_BPD, k_BPD, v_BPD = F.linear(x_BPD, attention.in_proj_weight, attention.in_proj_bias).chunk(3, -1)
    q_BJPE, k_BJPE, v_BJPE = [rearrange(t, "b p (j e) -> b j p e", j=2) for t in (q_BPD, k_BPD, v_BPD)]
    weights_BJPP = (q_BJPE @ k_BJPE.transpose(-1, -2) / 2).softmax(-1)
    output_BJPE = weights_BJPP @ v_BJPE
    # Independent formula: remove the projection along each token's own value.
    expected_BJPE = output_BJPE - v_BJPE * ((output_BJPE * v_BJPE).sum(-1, keepdim=True)
                                           / v_BJPE.square().sum(-1, keepdim=True))
    expected_BPD = attention.out_proj(rearrange(expected_BJPE, "b j p e -> b p (j e)"))
    actual_BPD, weights = xsa(x_BPD, x_BPD, x_BPD, need_weights=False)
    assert weights is None
    torch.testing.assert_close(actual_BPD, expected_BPD)
    expected_grad = torch.autograd.grad(expected_BPD.square().mean(), x_BPD, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual_BPD.square().mean(), x_BPD)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert xsa.in_proj_weight is attention.in_proj_weight
    assert xsa.out_proj is attention.out_proj


def test_single_token_cannot_write_its_own_value():
    attention = nn.MultiheadAttention(8, 2, batch_first=True)
    with torch.no_grad():
        attention.out_proj.bias.fill_(0.25)
    x_BPD = torch.randn(2, 1, 8)
    result_BPD, _ = ExclusiveSelfAttention(attention)(x_BPD, x_BPD, x_BPD, need_weights=False)
    torch.testing.assert_close(result_BPD, attention.out_proj.bias.expand_as(result_BPD), atol=1e-6, rtol=0)


def test_xsa_only_replaces_loop_self_attention_and_preserves_initialization():
    torch.manual_seed(0)
    reference = DeepSupervisedLoopedDiTPolicy(**TINY, **SPLIT, n_loops=3)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(0)
    policy = LoopedDiTXSAPolicy(**TINY, **SPLIT, n_loops=3)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert list(policy.state_dict()) == list(reference.state_dict())
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key])
    assert sum(p.numel() for p in policy.parameters()) == sum(p.numel() for p in reference.parameters())
    assert [isinstance(b.attn, ExclusiveSelfAttention) for b in policy.blocks] == [False, True, True, False]
    assert all(isinstance(b.cross, nn.MultiheadAttention) for b in policy.blocks)


def test_loop_and_decoder_paths_use_xsa_and_receive_gradients():
    policy = LoopedDiTXSAPolicy(**TINY, **SPLIT, n_loops=3)
    activate(policy)
    counts = [Mock(return_value=None) for _ in policy.blocks]
    handles = [b.attn.register_forward_hook(hook) for b, hook in zip(policy.blocks, counts)]
    b = batch()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    out = update(compose(), policy, optimizer, b)
    assert [counter.call_count for counter in counts] == [1, 3, 3, 3]
    assert set(out) == {"loss", "loss_loop_1", "loss_loop_2", "loss_loop_3", "grad_norm"}
    torch.testing.assert_close(out["loss"], out["loss_loop_3"] + (out["loss_loop_1"] + out["loss_loop_2"]) / 2)
    for block in policy.blocks[1:3]:
        assert torch.isfinite(block.attn.in_proj_weight.grad).all()
        assert block.attn.in_proj_weight.grad.abs().sum() > 0
    for counter in counts:
        counter.reset_mock()
    ctx_BLD = policy.encode(b)
    with torch.no_grad():
        final_BPA = policy.denoise(b["action"], torch.tensor([2, 7]), ctx_BLD)
    assert [counter.call_count for counter in counts] == [1, 3, 3, 1]
    torch.testing.assert_close(final_BPA, policy.denoise_loops(b["action"], torch.tensor([2, 7]), ctx_BLD)[-1])
    for handle in handles:
        handle.remove()


def test_config_sampler_and_hub_roundtrip(tmp_path):
    cfg = compose(["policy=dit_looped_deepsup_xsa", "data.action_horizon=4", "policy.d_model=16", "policy.n_heads=2",
                   "policy.n_train_steps=10", "policy.n_inference_steps=5", "policy.pretrained_vision=false",
                   "policy.lang_model=mock"])
    policy = instantiate(cfg.policy).eval()
    activate(policy)
    assert (policy.n_pre_layers, policy.n_loop_layers, policy.n_post_layers, policy.n_loops) == (2, 2, 2, 2)
    policy.save_pretrained(tmp_path)
    loaded = LoopedDiTXSAPolicy.from_pretrained(tmp_path).eval()
    assert isinstance(loaded.blocks[2].attn, ExclusiveSelfAttention)
    b = batch()
    with torch.no_grad():
        torch.manual_seed(0)
        expected_BPA = policy.predict_action(b)
        torch.manual_seed(0)
        actual_BPA = loaded.predict_action(b)
    torch.testing.assert_close(actual_BPA, expected_BPA)
    assert actual_BPA.shape == (2, 4, 7) and actual_BPA.abs().max() <= 1
