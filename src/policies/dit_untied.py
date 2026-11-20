"""Depth-matched DiT control with independent parameters at each refinement stage."""
from src.policies.dit_looped_deepsup import DeepSupervisedLoopedDiTPolicy


class UntiedDiTPolicy(DeepSupervisedLoopedDiTPolicy):
    def __init__(self, action_horizon: int, obs_horizon: int, d_model: int = 256, n_heads: int = 4,
                 n_pre_layers: int = 2, n_loop_layers: int = 2, n_post_layers: int = 2, n_loops: int = 4,
                 n_train_steps: int = 100, n_inference_steps: int = 100, pretrained_vision: bool = True,
                 lang_model: str = "openai/clip-vit-base-patch32"):
        # Allocate a separate set of middle blocks for each stage. Losses and the decoder stay shared.
        super().__init__(action_horizon=action_horizon, obs_horizon=obs_horizon, d_model=d_model, n_heads=n_heads,
                         n_pre_layers=n_pre_layers, n_loop_layers=n_loop_layers * n_loops,
                         n_post_layers=n_post_layers, n_loops=n_loops, n_train_steps=n_train_steps,
                         n_inference_steps=n_inference_steps, pretrained_vision=pretrained_vision, lang_model=lang_model)
        self.n_stage_layers = n_loop_layers

    def _refinement_stage(self, stage):
        start = self.n_pre_layers + stage * self.n_stage_layers
        return self.blocks[start:start + self.n_stage_layers]
