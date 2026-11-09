"""Deep-supervised looped DiT with exclusive self-attention in refinement blocks.
Dims: B batch, J attention heads, P action tokens, E head width, D model width.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.policies.dit_looped_deepsup import DeepSupervisedLoopedDiTPolicy


def exclusive_projection(output_BJPE, value_BJPE):
    """Remove each head output's component along that token's own value, before output projection."""
    # Accumulate in at least float32 so normalization stays safe for half-precision values.
    output_dtype = output_BJPE.dtype
    dtype = torch.promote_types(output_BJPE.dtype, torch.float32)
    output_BJPE = output_BJPE.to(dtype)
    direction_BJPE = F.normalize(value_BJPE.to(dtype), dim=-1, eps=1e-6)
    return (output_BJPE - direction_BJPE * (output_BJPE * direction_BJPE).sum(-1, keepdim=True)).to(output_dtype)


class ExclusiveSelfAttention(nn.Module):
    """Reuse a DiT self-attention module's parameters; modulate individual heads after SDPA."""

    def __init__(self, attention):
        super().__init__()
        self.num_heads = attention.num_heads
        self.in_proj_weight, self.in_proj_bias = attention.in_proj_weight, attention.in_proj_bias
        self.out_proj = attention.out_proj

    def forward(self, query, key, value, need_weights=False):
        if query is not key or query is not value or need_weights:
            raise ValueError("XSA supports self-attention with need_weights=False only.")
        q_BPD, k_BPD, v_BPD = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, -1)
        q_BJPE, k_BJPE, v_BJPE = [rearrange(t, "b p (j e) -> b j p e", j=self.num_heads)
                                  for t in (q_BPD, k_BPD, v_BPD)]
        output_BJPE = F.scaled_dot_product_attention(q_BJPE, k_BJPE, v_BJPE)
        output_BJPE = exclusive_projection(output_BJPE, v_BJPE)
        return self.out_proj(rearrange(output_BJPE, "b j p e -> b p (j e)")), None


class LoopedDiTXSAPolicy(DeepSupervisedLoopedDiTPolicy):
    def __init__(self, action_horizon: int, obs_horizon: int, d_model: int = 256, n_heads: int = 4,
                 n_pre_layers: int = 2, n_loop_layers: int = 2, n_post_layers: int = 2, n_loops: int = 2,
                 n_train_steps: int = 100, n_inference_steps: int = 100, pretrained_vision: bool = True,
                 lang_model: str = "openai/clip-vit-base-patch32"):
        super().__init__(action_horizon=action_horizon, obs_horizon=obs_horizon, d_model=d_model, n_heads=n_heads,
                         n_pre_layers=n_pre_layers, n_loop_layers=n_loop_layers, n_post_layers=n_post_layers,
                         n_loops=n_loops, n_train_steps=n_train_steps, n_inference_steps=n_inference_steps,
                         pretrained_vision=pretrained_vision, lang_model=lang_model)
        # Wrapping existing parameters preserves both initialization and checkpoint keys.
        for block in self.blocks[n_pre_layers:n_pre_layers + n_loop_layers]:
            block.attn = ExclusiveSelfAttention(block.attn)
