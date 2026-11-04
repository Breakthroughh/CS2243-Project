"""Torch port vs released RT-1-X outputs (fp32, eval mode).
usage: uv run python -m scripts.tmp_vla-baselines.rt1x_parity <dir with rt1x_{params,ref}.npz> [x64]"""
import sys

import numpy as np
import torch
from einops import rearrange

from scripts.convert_rt1x import load_rt1x
from src.policies.rt1x import I, K

d, *x64 = sys.argv[1:]
ref = dict(np.load(f"{d}/rt1x_ref{'_x64' * bool(x64)}.npz"))
policy = load_rt1x(f"{d}/rt1x_params.npz").eval()
if x64:
    policy = policy.double()


def report(name, ours, theirs):
    ours, theirs = ours.detach().numpy(), np.asarray(theirs)
    print(f"{name:22s} {str(ours.shape):20s} max|diff| {np.abs(ours - theirs).max():.2e}  max|ref| {np.abs(theirs).max():.2e}")


with torch.no_grad():
    x_BOCHW = rearrange(torch.from_numpy(ref["image"]), "b o h w c -> b o c h w")
    ctx_BE = torch.from_numpy(ref["lang"][:, 0])
    tokens = policy.tokenizer(x_BOCHW[0], ctx_BE.expand(x_BOCHW.shape[1], -1))
    report("image tokens", tokens, ref["image_tokens"][0])
    logits_BNU = policy.token_logits(x_BOCHW, ctx_BE)
    report("logits (all tokens)", logits_BNU, ref["logits"])
    ours = rearrange(logits_BNU, "b (o n) u -> b o n u", o=15)[:, -1, I - 1:-1].argmax(-1)
    theirs = rearrange(torch.from_numpy(ref["logits"]), "b (o n) u -> b o n u", o=15)[:, -1, I - 1:-1].argmax(-1)
    print("action tokens (last step) equal:", torch.equal(ours, theirs), ours.tolist())
