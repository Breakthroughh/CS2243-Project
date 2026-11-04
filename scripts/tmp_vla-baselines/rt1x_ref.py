"""Run in the throwaway JAX env: official RT-1-X (OXE models/rt1.py, released config) on fixed inputs -> params + outputs.
usage: python rt1x_ref.py <oxe_src> <checkpoint> <out_dir> [x64]
x64: run everything in float64 (fp32 is ill-conditioned on noise images: TokenLearner's softmax saturates)."""
import functools
import sys

import flax.serialization
import jax
import numpy as np
from flax.traverse_util import flatten_dict

oxe_src, ckpt_path, out, *x64 = sys.argv[1:]
sys.path.insert(0, f"{oxe_src}/models")
import efficientnet  # noqa: E402
import rt1  # noqa: E402

if x64:
    jax.config.update("jax_enable_x64", True)
    efficientnet.EfficientNetWithFilm = functools.partial(efficientnet.EfficientNetWithFilm, dtype=jax.numpy.float64)

ckpt = flax.serialization.msgpack_restore(open(ckpt_path, "rb").read())
dtype = np.float64 if x64 else np.float32
variables = jax.tree_util.tree_map(lambda a: np.asarray(a, dtype), {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]})
model = rt1.RT1(num_image_tokens=81, num_action_tokens=11, layer_size=256, vocab_size=512, use_token_learner=True,
                world_vector_range=(-2.0, 2.0))
flat = {f"params/{k}": v for k, v in flatten_dict(ckpt["params"], sep="/").items()}
flat |= {f"batch_stats/{k}": v for k, v in flatten_dict(ckpt["batch_stats"], sep="/").items()}
if not x64:
    np.savez(f"{out}/rt1x_params.npz", **{k: np.asarray(v) for k, v in flat.items()})

rng = np.random.default_rng(0)
O = 15
image = rng.random((1, O, 300, 300, 3)).astype(dtype)
lang = np.repeat(rng.standard_normal((1, 1, 512)).astype(dtype), O, 1)
lang /= np.linalg.norm(lang, axis=-1, keepdims=True)  # USE embeddings are unit norm
kw = dict(train=False, rngs={"random": jax.random.PRNGKey(0)})
image_tokens = model.apply(variables, image, lang, method=rt1.RT1.tokenize_image, **kw)
logits = model.apply(variables, {"image": image, "natural_language_embedding": lang}, act=None,
                     act_tokens=np.zeros((1, O, 11)), **kw)
np.savez(f"{out}/rt1x_ref{'_x64' * bool(x64)}.npz", image=image, lang=lang, image_tokens=np.asarray(image_tokens), logits=np.asarray(logits))
print(image_tokens.dtype, image_tokens.shape, logits.shape)
