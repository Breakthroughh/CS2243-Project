"""CPU contracts for loop-state capture and tokenwise convergence metrics."""
import pytest
import torch
from torch import nn

from analysis import capture_loops, convergence


class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, context, timestep):
        return x + self.value


class ToyPolicy(nn.Module):
    def __init__(self, loops):
        super().__init__()
        self.n_pre_layers, self.n_loop_layers, self.n_loops = 1, 2, loops
        self.blocks = nn.ModuleList([Add(10), Add(1), Add(2), Add(1000)])

    def denoise(self, x, timestep, context):
        x = self.blocks[0](x, context, timestep)
        for _ in range(self.n_loops):
            for block in self.blocks[1:3]:
                x = block(x, context, timestep)
        return self.blocks[3](x, context, timestep)


@pytest.mark.parametrize("loops", [2, 4, 6])
def test_capture_uses_real_forward_and_excludes_decoder(loops):
    policy = ToyPolicy(loops).eval()
    x = torch.zeros(2, 3, 4)
    before = policy.denoise(x, None, None)
    states = capture_loops(policy, x, None, None)
    assert states.shape == (loops + 1, 2, 3, 4)
    torch.testing.assert_close(states[:, 0, 0, 0], 10 + 3 * torch.arange(loops + 1).float())
    torch.testing.assert_close(policy.denoise(x, None, None), before)
    assert not any(b._forward_hooks or b._forward_pre_hooks for b in policy.blocks)
    assert not states.requires_grad


def test_capture_removes_hooks_when_forward_fails():
    policy = ToyPolicy(2)
    def fail(*args):
        raise RuntimeError("test failure")
    policy.blocks[2].forward = fail
    with pytest.raises(RuntimeError, match="test failure"):
        capture_loops(policy, torch.zeros(1, 2, 3), None, None)
    assert not any(b._forward_hooks or b._forward_pre_hooks for b in policy.blocks)


def test_cosine_is_per_token_then_averaged_not_flattened():
    states = torch.tensor([[[[1., 0.], [0., 100.]]], [[[1., 0.], [100., 0.]]]])
    result = convergence(states)
    torch.testing.assert_close(result["cosine"][:, 0], torch.tensor([0.5, 1.]))
    assert result["cosine"].shape == (2, 1)


def test_radial_change_is_visible_even_when_cosine_is_one():
    states = torch.tensor([[[[1., 0.]]], [[[2., 0.]]]])
    result = convergence(states)
    torch.testing.assert_close(result["cosine"][:, 0], torch.ones(2))
    torch.testing.assert_close(result["relative_l2"][:, 0], torch.tensor([0.5, 0.]))


def test_zero_vectors_have_undefined_cosine():
    result = convergence(torch.zeros(3, 2, 4, 8))
    assert torch.isnan(result["cosine"]).all()
