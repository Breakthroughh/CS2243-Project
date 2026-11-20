"""Shared-weight DiT refinement with final-loop epsilon loss.
Dims: B batch, P action horizon, A action (7), L context tokens, D model width.
"""
from src.policies.dit import DiTPolicy, modulate, timestep_embedding


class LoopedDiTPolicy(DiTPolicy):
    def __init__(self, action_horizon: int, obs_horizon: int, d_model: int = 256, n_heads: int = 4,
                 n_pre_layers: int = 2, n_loop_layers: int = 2, n_post_layers: int = 2, n_loops: int = 2,
                 n_train_steps: int = 100, n_inference_steps: int = 100, pretrained_vision: bool = True,
                 lang_model: str = "openai/clip-vit-base-patch32"):
        for name, value, minimum in (("n_pre_layers", n_pre_layers, 0), ("n_loop_layers", n_loop_layers, 1),
                                     ("n_post_layers", n_post_layers, 0), ("n_loops", n_loops, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        super().__init__(action_horizon=action_horizon, obs_horizon=obs_horizon, d_model=d_model,
                         n_layers=n_pre_layers + n_loop_layers + n_post_layers, n_heads=n_heads,
                         n_train_steps=n_train_steps, n_inference_steps=n_inference_steps,
                         pretrained_vision=pretrained_vision, lang_model=lang_model)
        self.n_pre_layers, self.n_loop_layers = n_pre_layers, n_loop_layers
        self.n_post_layers, self.n_loops = n_post_layers, n_loops

    def _refinement_stage(self, stage):
        return self.blocks[self.n_pre_layers:self.n_pre_layers + self.n_loop_layers]

    def denoise(self, x_BPA, k_B, ctx_BLD):
        """Refine hidden states at fixed noise level/context, then decode once to epsilon [B,P,A]."""
        k_BD = self.step_mlp(timestep_embedding(k_B, self.action_emb_PD.shape[-1]))
        h_BPD = self.action_in(x_BPA) + self.action_emb_PD
        loop_end = self.n_pre_layers + self.n_loop_layers
        for block in self.blocks[:self.n_pre_layers]:
            h_BPD = block(h_BPD, ctx_BLD, k_BD)
        for stage in range(self.n_loops):
            for block in self._refinement_stage(stage):
                h_BPD = block(h_BPD, ctx_BLD, k_BD)
        for block in self.blocks[loop_end:]:
            h_BPD = block(h_BPD, ctx_BLD, k_BD)
        shift, scale = self.final_ada(k_BD)[:, None].chunk(2, -1)
        return self.action_out(modulate(self.final_norm(h_BPD), shift, scale))
