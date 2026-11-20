"""Untied control contract. Dims: R refinement stages, B batch, P action horizon, A action, L context, D width."""
import torch
from hydra.utils import instantiate
from torch import nn

from src.config import compose
from src.policies.dit_looped_deepsup import DeepSupervisedLoopedDiTPolicy
from src.policies import dit_untied
from tests.test_dit_looped import TINY, activate, mock_encoders  # noqa: F401


def test_config_selects_twelve_distinct_blocks():
    cfg = compose(["policy=dit_untied", "policy.d_model=16", "policy.pretrained_vision=false"])
    policy = instantiate(cfg.policy)
    assert isinstance(policy, dit_untied.UntiedDiTPolicy)
    assert len(policy.blocks) == 12
    assert all(isinstance(block.attn, nn.MultiheadAttention) for block in policy.blocks)
    assert len({id(block.attn.in_proj_weight) for block in policy.blocks}) == 12


def test_untied_control_matches_tied_computation_when_weights_match():
    tied = DeepSupervisedLoopedDiTPolicy(**TINY, n_loops=4).eval()
    untied = dit_untied.UntiedDiTPolicy(**TINY, n_loops=4).eval()
    activate(tied)
    tied_state = tied.state_dict()
    source_blocks = [0, 1] + [2, 3] * 4 + [4, 5]
    for key in untied.state_dict():
        pieces = key.split(".")
        if pieces[0] == "blocks":
            pieces[1] = str(source_blocks[int(pieces[1])])
        untied.state_dict()[key].copy_(tied_state[".".join(pieces)])

    x_BPA, k_B, ctx_BLD = torch.randn(2, 4, 7), torch.tensor([2, 7]), torch.randn(2, 5, 16)
    with torch.no_grad():
        predictions_RBPA = tied.denoise_loops(x_BPA, k_B, ctx_BLD)
        torch.testing.assert_close(untied.denoise_loops(x_BPA, k_B, ctx_BLD), predictions_RBPA)
        torch.testing.assert_close(untied.denoise(x_BPA, k_B, ctx_BLD), predictions_RBPA[-1])
