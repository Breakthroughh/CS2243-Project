"""Torch port vs official Octo outputs, stage by stage (fp32, eval mode).
usage: uv run python scripts/tmp_vla-baselines/octo_parity.py <dir with octo_{size}_{params,ref}.npz> {small,base}"""
import sys

import numpy as np
import torch
from einops import rearrange

from scripts.convert_octo import load_octo
from src.policies.octo import encode_language

d, size = sys.argv[1:]
ref = dict(np.load(f"{d}/octo_{size}_ref.npz"))
policy = load_octo(f"{d}/octo_{size}_params.npz", size).eval()


def report(name, ours, theirs):
    ours, theirs = ours.detach().numpy(), np.asarray(theirs)
    err = np.abs(ours - theirs).max()
    print(f"{name:24s} {str(ours.shape):22s} max|diff| {err:.2e}  max|ref| {np.abs(theirs).max():.2e}")
    return err


with torch.no_grad():
    lang_BLE = encode_language([str(ref["text"])], "cpu")
    for O in (1, 2):
        views = {name: rearrange(torch.from_numpy(ref[f"O{O}_image_{name}"]).float() / 255, "b o h w c -> b o c h w")
                 for name in ("primary", "wrist")}
        report(f"O{O} t5 tokens", lang_BLE, ref[f"O{O}_lang"])
        for name, x in views.items():
            x = rearrange(x, "b o c h w -> (b o) c h w")
            stem = policy.stems[name](torch.cat([x * 2 - 1, torch.full_like(x, -1)], 1))
            report(f"O{O} stem {name}", stem, ref[f"O{O}_stem_{name}"].reshape(stem.shape))
        lang = torch.from_numpy(ref[f"O{O}_lang"])  # isolate transformer from T5 differences
        emb_BOD = policy.encode(views, lang)
        report(f"O{O} readout", emb_BOD, ref[f"O{O}_readout"][:, :, 0])
        eps = policy.head(torch.from_numpy(ref[f"O{O}_readout"][:, :, 0]), torch.from_numpy(ref[f"O{O}_noisy"]),
                          torch.from_numpy(ref[f"O{O}_time"]))
        report(f"O{O} eps (ref readout)", eps, ref[f"O{O}_eps"])
        report(f"O{O} eps (end to end)", policy.head(emb_BOD, torch.from_numpy(ref[f"O{O}_noisy"]),
                                                     torch.from_numpy(ref[f"O{O}_time"])), ref[f"O{O}_eps"])
