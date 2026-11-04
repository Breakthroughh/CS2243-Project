"""RT-1-X (Open X-Embodiment, 2023) in torch, mirroring google-deepmind/open_x_embodiment models/rt1.py with the
released RT-1-X config (rt1_inference_example.py): 15 frames, 81 TokenLearner tokens, key dim 256, vocab 512,
SwiGLU FFN, world vector +-2. Language: frozen all-MiniLM-L6-v2 sentence embedding as in BAKU (Haldar et al. 2024),
in place of the Universal Sentence Encoder.
Dims: B batch, O obs window, I image tokens / frame (81), K action tokens (11), N tokens O * (I + K), D width (512),
U vocab (512), E lang width (384), S spatial positions, C channels, H height, W width, A action (7)."""
import functools
import math

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from torch import nn

from src.policies.baseline import Policy, lecun_init_

I, K, D, VOCAB, RES = 81, 11, 512, 512, 300
# tokens 1..10: world vector (3), rotation (3), gripper, base yaw, base xy (2); token 0 = mode (1: arm + gripper)
LOW = torch.tensor([-2.0] * 3 + [-math.pi / 2] * 3 + [-1.0, -math.pi, -1.0, -1.0])
# libero deltas in [-1, 1] -> rt-1 units, using the OXE colab's per-dataset rescale targets (world 1.75, rotation 1.4)
SCALE_A = torch.tensor([1.75] * 3 + [1.4] * 3 + [1.0])
_LANG = {}


@functools.cache
def _minilm():
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()


def encode_language(texts, device):
    """Frozen all-MiniLM-L6-v2: mean-pooled, L2-normalised [B, E]. Cached per instruction, kept out of the state_dict."""
    missing = [t for t in dict.fromkeys(texts) if t not in _LANG]
    if missing:
        tokenizer, encoder = _minilm()
        ids = tokenizer(missing, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h_BLE, m_BL1 = encoder(**ids).last_hidden_state, ids["attention_mask"][..., None]
        _LANG.update(zip(missing, F.normalize((h_BLE * m_BL1).sum(1) / m_BL1.sum(1), dim=-1)))
    return torch.stack([_LANG[t] for t in texts]).to(device)


def tokenize(action_BA, vocab=VOCAB):
    """libero action [..., A] -> rt-1 tokens [..., K]: clip to bounds, int((x - low) / (high - low) * (vocab - 1))."""
    x = torch.cat([action_BA * SCALE_A.to(action_BA), torch.zeros_like(action_BA[..., :3])], -1)
    low = LOW.to(x)
    bins = ((x.clamp(low, -low) - low) / (-2 * low) * (vocab - 1)).long()
    return torch.cat([torch.ones_like(bins[..., :1]), bins], -1)


def detokenize(tokens_BK, vocab=VOCAB):
    low = LOW[:7].to(tokens_BK.device)
    return (tokens_BK[..., 1:8].float() / (vocab - 1) * (-2 * low) + low) / SCALE_A.to(low)


def attention_mask(n_image, n_action, window):
    """[N, N], True = may attend: causal, and only image tokens are attended (action tokens are zero placeholders)."""
    is_image_N = torch.tensor(([True] * n_image + [False] * n_action) * window)
    return torch.ones(len(is_image_N), len(is_image_N), dtype=torch.bool).tril() & is_image_N[None, :]


def pad_crop(x, pad=(10, 20)):
    """RT-1-X train augmentation: zero-pad then random crop back = random shift of up to (pad_h, pad_w) px.
    The colab pads (20, 50) px at 256 x 320; scaled to libero's 128 x 128."""
    N, _, H, W = x.shape
    shift = lambda p, n: torch.randint(-p, p + 1, (N,), device=x.device) * 2 / n
    theta = torch.zeros(N, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = 1, 1, shift(pad[1], W), shift(pad[0], H)
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode="nearest", align_corners=False)


class FiLM(nn.Module):
    """(1 + mul(ctx)) * x + add(ctx), zero-initialised (identity at init)."""

    def __init__(self, ctx, channels):
        super().__init__()
        self.add, self.mul = nn.Linear(ctx, channels), nn.Linear(ctx, channels)

    def forward(self, x_BCHW, ctx_BE):
        return (1 + self.mul(ctx_BE)[..., None, None]) * x_BCHW + self.add(ctx_BE)[..., None, None]


class TokenLearner(nn.Module):
    """v1.1: LN -> MLP(64, gelu) -> softmax over S per token -> weighted sum. [B, S, D] -> [B, I, D]."""

    def __init__(self, d, n_tokens, hidden=64):
        super().__init__()
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, n_tokens))

    def forward(self, x_BSD):
        w_BIS = self.mlp(self.norm(x_BSD)).transpose(1, 2).softmax(-1)
        return torch.einsum("bis,bsd->bid", w_BIS, x_BSD)


class TFSamePad(nn.Module):
    """TF 'SAME' padding around a strided conv (extra pixel bottom/right); torchvision pads symmetrically."""

    def __init__(self, conv):
        super().__init__()
        self.conv, conv.padding = conv, (0, 0)

    def forward(self, x):
        (kh, kw), (sh, sw) = self.conv.kernel_size, self.conv.stride
        ph = max((math.ceil(x.shape[-2] / sh) - 1) * sh + kh - x.shape[-2], 0)
        pw = max((math.ceil(x.shape[-1] / sw) - 1) * sw + kw - x.shape[-1], 0)
        return self.conv(F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)))


def tf_efficientnet_b3(pretrained):
    """torchvision EfficientNet-B3 with TF numerics: SAME padding on strided convs, BatchNorm eps 1e-3 / momentum .99."""
    net = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
    for name, m in list(net.named_modules()):
        if isinstance(m, nn.Conv2d) and m.stride != (1, 1):
            parent, child = name.rsplit(".", 1)
            setattr(net.get_submodule(parent), child, TFSamePad(m))
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.01
    return net


class ImageTokenizer(nn.Module):
    """EfficientNet-B3 with FiLM after each of its 26 MBConv blocks -> 1x1 conv (1536 -> D) -> FiLM -> TokenLearner.
    [B, C, 300, 300] in [0, 1], ctx [B, E] -> [B, I, D]."""

    def __init__(self, pretrained, ctx=384):
        super().__init__()
        net = tf_efficientnet_b3(pretrained)
        self.stem, self.top = net.features[0], net.features[8]
        self.blocks = nn.ModuleList(block for stage in net.features[1:8] for block in stage)
        self.film = nn.ModuleList(FiLM(ctx, b.out_channels) for b in self.blocks)
        self.conv = nn.Conv2d(net.features[8][0].out_channels, D, 1, bias=False)
        self.film_out = FiLM(ctx, D)
        self.token_learner = TokenLearner(D, I)
        self.register_buffer("mean_C", torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer("std_C", torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    def forward(self, x_BCHW, ctx_BE):
        x = self.stem((x_BCHW - self.mean_C) / self.std_C)
        for block, film in zip(self.blocks, self.film):
            x = film(block(x), ctx_BE)
        x = self.film_out(self.conv(self.top(x)), ctx_BE)
        return self.token_learner(rearrange(x, "b d h w -> b (h w) d"))


class Block(nn.Module):
    """Pre-LN: x + attn(LN x) (8 heads x key dim 256, attention dropout); x + dropout(SwiGLU(LN x)), no biases in FFN."""

    def __init__(self, heads=8, key=256, ff=512, dropout=0.1):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(D, eps=1e-6), nn.LayerNorm(D, eps=1e-6)
        self.qkv, self.o = nn.Linear(D, 3 * heads * key), nn.Linear(heads * key, D)
        self.up, self.gate, self.down = nn.Linear(D, ff, bias=False), nn.Linear(D, ff, bias=False), nn.Linear(ff, D, bias=False)

    def forward(self, x_BND, allow_NN):
        q, k, v = rearrange(self.qkv(self.norm1(x_BND)), "b n (three h e) -> three b h n e", three=3, h=self.heads)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allow_NN, dropout_p=self.dropout if self.training else 0.0)
        x_BND = x_BND + self.o(rearrange(a, "b h n e -> b n (h e)"))
        y = self.norm2(x_BND)
        return x_BND + F.dropout(self.down(F.silu(self.up(y)) * self.gate(y)), self.dropout, self.training)


class RT1XPolicy(Policy):
    def __init__(self, obs_horizon: int = 15, pretrained: bool = True, layers: int = 8, lang_dim: int = 384):
        """lang_dim: 384 = MiniLM (from scratch); 512 = USE, needed to load the released RT-1-X FiLM weights."""
        super().__init__()
        self.O = obs_horizon
        self.tokenizer = ImageTokenizer(pretrained, ctx=lang_dim)
        self.inp = nn.Linear(D, D)
        self.pos = nn.Linear(obs_horizon * (I + K), D)  # Dense over one_hot(position)
        self.blocks = nn.ModuleList(Block() for _ in range(layers))
        self.logits = nn.Linear(D, VOCAB)
        backbone = {id(p) for m in (self.tokenizer.stem, self.tokenizer.blocks, self.tokenizer.top) for p in m.parameters()}
        for m in self.modules():  # flax lecun init everywhere except the (ImageNet) EfficientNet
            if isinstance(m, (nn.Linear, nn.Conv2d)) and id(m.weight) not in backbone:
                lecun_init_(m)
        for film in (*self.tokenizer.film, self.tokenizer.film_out):
            for w in (film.add.weight, film.add.bias, film.mul.weight, film.mul.bias):
                nn.init.zeros_(w)

    def views(self, obs):
        """libero obs -> primary camera [B, O, C, 300, 300] in [0, 1] (flipped upright, shift-augmented if training)."""
        image = obs["image"][:, :, 0].flip(-3)  # libero renders upside down
        x = rearrange(image, "b o h w c -> (b o) c h w").float() / 255
        if self.training:
            x = pad_crop(x)
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True)
        return rearrange(x, "(b o) c h w -> b o c h w", b=image.shape[0])

    def token_logits(self, x_BOCHW, ctx_BE):
        """-> logits [B, N, U] over the full token sequence."""
        B, O = x_BOCHW.shape[:2]
        ctx = repeat(ctx_BE, "b e -> (b o) e", o=O)
        image_BOID = rearrange(self.tokenizer(rearrange(x_BOCHW, "b o c h w -> (b o) c h w"), ctx), "(b o) i d -> b o i d", b=B)
        tokens_BOND = torch.cat([image_BOID, image_BOID.new_zeros(B, O, K, D)], 2)
        x_BND = self.inp(rearrange(tokens_BOND, "b o n d -> b (o n) d")) + self.pos.weight.T + self.pos.bias
        allow_NN = attention_mask(I, K, O).to(x_BND.device)
        for block in self.blocks:
            x_BND = block(x_BND, allow_NN)
        return self.logits(x_BND)

    def action_logits(self, obs):
        """-> logits [B, O, K, U]: action k at step t is read off the token before it (last image token for k = 0)."""
        logits_BNU = self.token_logits(self.views(obs), encode_language(obs["lang"], obs["image"].device))
        return rearrange(logits_BNU, "b (o n) u -> b o n u", o=self.O)[:, :, I - 1:-1]

    def compute_loss(self, batch):
        """Cross-entropy over all O x K action tokens; batch["action"] [B, O, A] is aligned with the obs window."""
        logits_BOKU = self.action_logits(batch)
        return {"loss": F.cross_entropy(rearrange(logits_BOKU, "b o k u -> (b o k) u"), tokenize(batch["action"]).flatten())}

    @torch.no_grad()
    def predict_action(self, obs):
        return detokenize(self.action_logits(obs)[:, -1].argmax(-1))[:, None]
