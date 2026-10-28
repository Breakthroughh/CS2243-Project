"""DiT diffusion policy (DDPM, epsilon prediction). Action tokens self-attend and cross-attend to obs/lang context.
Dims: B batch, O obs horizon, P action horizon, V views (agentview, wrist), H height, W width, C rgb, S state (8),
A action (7), L context tokens (O*V image + O state + 1 lang), D model width, E image feature (512)."""
import math

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from src.policy import Policy

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def squaredcos_alpha_bar(n_steps):
    """-> alpha_bar [n_steps]; squaredcos_cap_v2 schedule (Nichol & Dhariwal), as in Diffusion Policy."""
    f = lambda t: math.cos((t / n_steps + 0.008) / 1.008 * math.pi / 2) ** 2
    beta = torch.tensor([min(1 - f(i + 1) / f(i), 0.999) for i in range(n_steps)])
    return torch.cumprod(1 - beta, 0)


def timestep_embedding(k_B, dim):
    """Sinusoidal embedding of diffusion step k -> [B, dim]."""
    freq = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=k_B.device) / (dim // 2))
    angle_BD = k_B[:, None].float() * freq[None]
    return torch.cat([angle_BD.cos(), angle_BD.sin()], -1)


def modulate(x_BPD, shift_B1D, scale_B1D):
    return x_BPD * (1 + scale_B1D) + shift_B1D


class DiTBlock(nn.Module):
    """adaLN-zero self-attention + cross-attention + MLP; the diffusion step conditions via adaLN."""

    def __init__(self, d, heads):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d, elementwise_affine=False) for _ in range(3))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 7 * d))
        nn.init.zeros_(self.ada[1].weight), nn.init.zeros_(self.ada[1].bias)

    def forward(self, x_BPD, ctx_BLD, k_BD):
        shift1, scale1, gate1, gate2, shift3, scale3, gate3 = self.ada(k_BD)[:, None].chunk(7, -1)
        h_BPD = modulate(self.norm1(x_BPD), shift1, scale1)
        x_BPD = x_BPD + gate1 * self.attn(h_BPD, h_BPD, h_BPD, need_weights=False)[0]
        x_BPD = x_BPD + gate2 * self.cross(self.norm2(x_BPD), ctx_BLD, ctx_BLD, need_weights=False)[0]
        return x_BPD + gate3 * self.mlp(modulate(self.norm3(x_BPD), shift3, scale3))


class DiTPolicy(Policy):
    def __init__(self, action_horizon: int, obs_horizon: int, d_model: int = 256, n_layers: int = 6, n_heads: int = 4,
                 n_train_steps: int = 100, n_inference_steps: int = 100, pretrained_vision: bool = True,
                 lang_model: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.P, self.O, self.n_inference_steps = action_horizon, obs_horizon, n_inference_steps
        d = d_model
        self.register_buffer("alpha_bar_K", squaredcos_alpha_bar(n_train_steps))  # K = n_train_steps

        # context encoders
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained_vision else None)
        resnet.fc = nn.Identity()
        self.vision = resnet  # shared across views
        self.image_proj = nn.Linear(512, d)
        self.state_proj = nn.Linear(8, d)
        self.tokenizer = CLIPTokenizer.from_pretrained(lang_model)
        self.lang = CLIPTextModel.from_pretrained(lang_model).requires_grad_(False)  # frozen
        self.lang_proj = nn.Linear(self.lang.config.hidden_size, d)
        self.view_emb_VD = nn.Parameter(torch.zeros(2, d))
        self.obs_emb_OD = nn.Parameter(torch.zeros(obs_horizon, d))
        self.register_buffer("mean_C", IMAGENET_MEAN)
        self.register_buffer("std_C", IMAGENET_STD)

        # denoiser
        self.action_in = nn.Linear(7, d)
        self.action_emb_PD = nn.Parameter(torch.zeros(action_horizon, d))
        self.step_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(DiTBlock(d, n_heads) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.action_out = nn.Linear(d, 7)
        for layer in (self.final_ada[1], self.action_out):
            nn.init.zeros_(layer.weight), nn.init.zeros_(layer.bias)
        for emb in (self.view_emb_VD, self.obs_emb_OD, self.action_emb_PD):
            nn.init.normal_(emb, std=0.02)

    def encode(self, obs):
        """obs -> context [B, L, D]."""
        B, O, V = obs["image"].shape[:3]
        x_NCHW = rearrange(obs["image"], "b o v h w c -> (b o v) c h w").float() / 255
        x_NCHW = (x_NCHW - self.mean_C[:, None, None]) / self.std_C[:, None, None]
        feat_BOVE = rearrange(self.vision(x_NCHW), "(b o v) e -> b o v e", b=B, o=O, v=V)
        image_BOVD = self.image_proj(feat_BOVE) + self.view_emb_VD + self.obs_emb_OD[:, None]
        state_BOD = self.state_proj(obs["state"]) + self.obs_emb_OD
        with torch.no_grad():
            tokens = self.tokenizer(obs["lang"], padding=True, truncation=True, return_tensors="pt").to(self.device)
            lang_BE = self.lang(**tokens).pooler_output
        lang_B1D = self.lang_proj(lang_BE)[:, None]
        return torch.cat([rearrange(image_BOVD, "b o v d -> b (o v) d"), state_BOD, lang_B1D], 1)

    def denoise(self, x_BPA, k_B, ctx_BLD):
        """Noisy actions at step k -> predicted noise [B, P, A]."""
        k_BD = self.step_mlp(timestep_embedding(k_B, self.action_emb_PD.shape[-1]))
        h_BPD = self.action_in(x_BPA) + self.action_emb_PD
        for block in self.blocks:
            h_BPD = block(h_BPD, ctx_BLD, k_BD)
        shift, scale = self.final_ada(k_BD)[:, None].chunk(2, -1)
        return self.action_out(modulate(self.final_norm(h_BPD), shift, scale))

    def compute_loss(self, batch):
        action_BPA = batch["action"]
        B = action_BPA.shape[0]
        k_B = torch.randint(0, len(self.alpha_bar_K), (B,), device=self.device)
        a_B11 = self.alpha_bar_K[k_B][:, None, None]
        noise_BPA = torch.randn_like(action_BPA)
        x_BPA = a_B11.sqrt() * action_BPA + (1 - a_B11).sqrt() * noise_BPA
        return {"loss": F.mse_loss(self.denoise(x_BPA, k_B, self.encode(batch)), noise_BPA)}

    def predict_action(self, obs):
        """DDPM ancestral sampling over n_inference_steps evenly strided steps (diffusers DDPMScheduler)."""
        ctx_BLD = self.encode(obs)
        B, K = ctx_BLD.shape[0], len(self.alpha_bar_K)
        x_BPA = torch.randn(B, self.P, 7, device=self.device)
        steps = list(range(0, K, K // self.n_inference_steps))[::-1]
        for i, k in enumerate(steps):
            a_bar = self.alpha_bar_K[k]
            a_bar_prev = self.alpha_bar_K[steps[i + 1]] if i + 1 < len(steps) else torch.ones_like(a_bar)
            beta = 1 - a_bar / a_bar_prev
            eps_BPA = self.denoise(x_BPA, torch.full((B,), k, device=self.device), ctx_BLD)
            x0_BPA = ((x_BPA - (1 - a_bar).sqrt() * eps_BPA) / a_bar.sqrt()).clamp(-1, 1)
            x_BPA = (a_bar_prev.sqrt() * beta * x0_BPA + (1 - beta).sqrt() * (1 - a_bar_prev) * x_BPA) / (1 - a_bar)
            if i + 1 < len(steps):
                x_BPA = x_BPA + (beta * (1 - a_bar_prev) / (1 - a_bar)).sqrt() * torch.randn_like(x_BPA)
        return x_BPA.clamp(-1, 1)
