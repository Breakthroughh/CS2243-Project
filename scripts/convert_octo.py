"""Official Octo 1.5 flax params -> src.policies.octo.OctoPolicy state_dict (T5 excluded: loaded from HF t5-base).
Params come from scripts/tmp_vla-baselines/octo_ref.py (flattened '/'-joined npz).
usage: uv run python scripts/convert_octo.py <octo_{size}_params.npz> {small,base} <out_dir>  (-> save_pretrained)"""
import sys

import numpy as np
import torch

from src.policies.octo import VIEWS, OctoPolicy


def octo_state_dict(flat, policy):
    T, H = "octo_transformer/", "heads_action/diffusion_model/"
    dense = lambda k: {"weight": flat[f"{k}/kernel"].T, "bias": flat[f"{k}/bias"]}
    norm = lambda k: {"weight": flat[f"{k}/scale"], "bias": flat[f"{k}/bias"]}
    conv = lambda k: {"weight": flat[f"{k}/kernel"].transpose(3, 2, 0, 1), "bias": flat[f"{k}/bias"]}
    sd = {}
    add = lambda prefix, d: sd.update({f"{prefix}.{k}": v for k, v in d.items()})
    for name in VIEWS:
        stem = f"{T}observation_tokenizers_{name}/SmallStem16_0/"
        for i in range(4):
            add(f"stems.{name}.convs.{i}", conv(f"{stem}StdConv_{i}"))
            add(f"stems.{name}.norms.{i}", norm(f"{stem}GroupNorm_{i}"))
        add(f"stems.{name}.embedding", conv(f"{stem}embedding"))
    for group in ("task_language", *(f"obs_{name}" for name in VIEWS)):
        add(f"proj.{group}", dense(f"{T}{group}_projection"))
    for group in policy.pos:
        sd[f"pos.{group}"] = flat[f"{T}{group}_pos_embedding"]
    for i in range(len(policy.transformer.layers)):
        blk, layer = f"{T}BlockTransformer_0/Transformer_0/encoderblock_{i}/", f"transformer.layers.{i}"
        attn = f"{blk}MultiHeadDotProductAttention_0/"
        qkv = [flat[f"{attn}{n}/kernel"].reshape(flat[f"{attn}{n}/kernel"].shape[0], -1).T for n in ("query", "key", "value")]
        sd[f"{layer}.self_attn.in_proj_weight"] = np.concatenate(qkv)
        sd[f"{layer}.self_attn.in_proj_bias"] = np.concatenate([flat[f"{attn}{n}/bias"].reshape(-1) for n in ("query", "key", "value")])
        sd[f"{layer}.self_attn.out_proj.weight"] = flat[f"{attn}out/kernel"].reshape(-1, flat[f"{attn}out/kernel"].shape[-1]).T
        sd[f"{layer}.self_attn.out_proj.bias"] = flat[f"{attn}out/bias"]
        add(f"{layer}.norm1", norm(f"{blk}LayerNorm_0"))
        add(f"{layer}.norm2", norm(f"{blk}LayerNorm_1"))
        add(f"{layer}.linear1", dense(f"{blk}MlpBlock_0/Dense_0"))
        add(f"{layer}.linear2", dense(f"{blk}MlpBlock_0/Dense_1"))
    add("transformer.norm", norm(f"{T}BlockTransformer_0/Transformer_0/encoder_norm"))
    sd["head.time.0.kernel"] = flat[f"{H}time_preprocess/kernel"]
    add("head.time.1", dense(f"{H}cond_encoder/Dense_0"))
    add("head.time.3", dense(f"{H}cond_encoder/Dense_1"))
    add("head.reverse.inp", dense(f"{H}reverse_network/Dense_0"))
    add("head.reverse.out", dense(f"{H}reverse_network/Dense_1"))
    for j in range(len(policy.head.reverse.ups)):
        res = f"{H}reverse_network/MLPResNetBlock_{j}/"
        add(f"head.reverse.norms.{j}", norm(f"{res}LayerNorm_0"))
        add(f"head.reverse.ups.{j}", dense(f"{res}Dense_0"))
        add(f"head.reverse.downs.{j}", dense(f"{res}Dense_1"))
    missing = {k for k, _ in policy.named_parameters()} - set(sd)
    assert not missing, f"unmapped params: {missing}"
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}


def load_octo(params_npz, size):
    policy = OctoPolicy(size=size)
    result = policy.load_state_dict(octo_state_dict(dict(np.load(params_npz)), policy), strict=False)
    assert not result.unexpected_keys and all("_K" in k or "_A" in k for k in result.missing_keys)  # buffers only
    return policy


if __name__ == "__main__":
    params_npz, size, out = sys.argv[1:]
    load_octo(params_npz, size).save_pretrained(out)
