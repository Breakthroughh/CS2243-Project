"""Run in the throwaway JAX env (see parity.md): official Octo 1.5 on fixed inputs -> params + per-stage outputs.
usage: python octo_ref.py <octo_src> <out_dir> {small,base}"""
import json
import sys

import flax.serialization
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

octo_src, out, size = sys.argv[1:]
sys.path.insert(0, octo_src)
from octo.model.octo_module import OctoModule  # noqa: E402

REVISIONS = {"small": "dc9aa3019f764726c770814b27e4ab0fc6e32a58", "base": "ee3c10e8edd6ce2e8b1e8744d3c6fba4097bed48"}
d = snapshot_download(f"rail-berkeley/octo-{size}-1.5", revision=REVISIONS[size], local_dir=f"{out}/octo-{size}-1.5")
params = flax.serialization.msgpack_restore(open(f"{d}/300000/default/checkpoint", "rb").read())
module = OctoModule.create(**json.load(open(f"{d}/config.json"))["model"])
bound = module.bind({"params": params})
np.savez(f"{out}/octo_{size}_params.npz", **{k: np.asarray(v) for k, v in flatten_dict(params, sep="/").items()
                                             if "hf_model" not in k})

text = "put the black bowl on the plate"
ids = AutoTokenizer.from_pretrained("t5-base")([text], padding="max_length", max_length=16, truncation=True, return_tensors="np")
rng = np.random.default_rng(0)
ref = {"input_ids": ids["input_ids"]}
for O in (1, 2):
    obs = {"image_primary": rng.integers(0, 256, (1, O, 256, 256, 3), dtype=np.uint8),
           "image_wrist": rng.integers(0, 256, (1, O, 128, 128, 3), dtype=np.uint8)}
    obs["pad_mask_dict"] = {k: np.ones((1, O), bool) for k in list(obs)}
    tasks = {"language_instruction": dict(ids), "pad_mask_dict": {"language_instruction": np.ones(1, bool)}}
    pad = np.ones((1, O), bool)
    tr = bound.octo_transformer
    outputs = tr(obs, tasks, pad, train=False)
    noisy = rng.standard_normal((1, O, 28)).astype(np.float32)
    time = rng.integers(0, 20, (1, O, 1))
    ref |= {f"O{O}_image_primary": obs["image_primary"], f"O{O}_image_wrist": obs["image_wrist"],
            f"O{O}_stem_primary": tr.observation_tokenizers["primary"](obs, tasks, train=False).tokens,
            f"O{O}_stem_wrist": tr.observation_tokenizers["wrist"](obs, tasks, train=False).tokens,
            f"O{O}_lang": tr.task_tokenizers["language"](obs, tasks, train=False).tokens,
            f"O{O}_readout": outputs["readout_action"].tokens, f"O{O}_noisy": noisy, f"O{O}_time": time,
            f"O{O}_eps": bound.heads["action"](outputs, time=jnp.asarray(time), noisy_actions=jnp.asarray(noisy), train=False)}
np.savez(f"{out}/octo_{size}_ref.npz", text=np.array(text), **{k: np.asarray(v) for k, v in ref.items()})
print({k: np.asarray(v).shape for k, v in ref.items()})
