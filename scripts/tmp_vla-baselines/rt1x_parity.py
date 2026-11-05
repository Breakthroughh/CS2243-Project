"""Torch port vs released RT-1-X outputs, float32 on both sides (official precision), eval mode.
usage: uv run python -m scripts.tmp_vla-baselines.rt1x_parity <dir with rt1x_params.npz, rt1x_ref[_frames].npz> [frames]"""
import sys

import numpy as np
import torch
from einops import rearrange

from scripts.convert_rt1x import load_rt1x
from src.policies.rt1x import I, K

d, *frames = sys.argv[1:]
ref = dict(np.load(f"{d}/rt1x_ref{'_frames' * bool(frames)}.npz"))
policy = load_rt1x(f"{d}/rt1x_params.npz").eval()


def report(name, ours, theirs):
    ours, theirs = ours.detach().numpy(), np.asarray(theirs)
    err = np.abs(ours - theirs)
    print(f"{name:22s} {str(ours.shape):20s} max|diff| {err.max():.2e}  rel {err.max() / np.abs(theirs).max():.1e}  "
          f"dtypes {ours.dtype}/{theirs.dtype}")


with torch.no_grad():
    x_BOCHW = rearrange(torch.from_numpy(ref["image"]), "b o h w c -> b o c h w")
    ctx_BE = torch.from_numpy(ref["lang"][:, 0])
    tokens = policy.tokenizer(x_BOCHW[0], ctx_BE.expand(x_BOCHW.shape[1], -1))
    report("image tokens", tokens, ref["image_tokens"][0])
    logits_BNU = policy.token_logits(x_BOCHW, ctx_BE)
    report("logits (all tokens)", logits_BNU, ref["logits"])
    ours = rearrange(logits_BNU, "b (o n) u -> b o n u", o=15)[:, :, I - 1:-1].argmax(-1)
    theirs = rearrange(torch.from_numpy(ref["logits"]), "b (o n) u -> b o n u", o=15)[:, :, I - 1:-1].argmax(-1)
    print(f"action tokens equal: {(ours == theirs).float().mean().item():.1%} of 15 steps x 11 tokens | "
          f"last step equal: {torch.equal(ours[:, -1], theirs[:, -1])} {ours[0, -1].tolist()}")
