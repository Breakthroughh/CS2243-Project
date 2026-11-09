"""Looped DiT with final-loop epsilon loss plus mean intermediate-loop loss.
Dims: R refinement loops, B batch, P action horizon, A action (7), L context tokens, D model width.
"""
import torch
import torch.nn.functional as F

from src.policies.dit import modulate, timestep_embedding
from src.policies.dit_looped import LoopedDiTPolicy


class DeepSupervisedLoopedDiTPolicy(LoopedDiTPolicy):
    # Inherit final-only denoise/predict_action: intermediate decoding is used only for loss.
    def denoise_loops(self, x_BPA, k_B, ctx_BLD):
        """Decode each loop through shared post-loop blocks/head -> epsilon [R,B,P,A]."""
        k_BD = self.step_mlp(timestep_embedding(k_B, self.action_emb_PD.shape[-1]))
        h_BPD = self.action_in(x_BPA) + self.action_emb_PD
        loop_end = self.n_pre_layers + self.n_loop_layers
        for block in self.blocks[:self.n_pre_layers]:
            h_BPD = block(h_BPD, ctx_BLD, k_BD)
        shift, scale = self.final_ada(k_BD)[:, None].chunk(2, -1)
        predictions = []
        for _ in range(self.n_loops):
            for block in self.blocks[self.n_pre_layers:loop_end]:
                h_BPD = block(h_BPD, ctx_BLD, k_BD)
            # The decoder branches off; the next refinement receives the pre-decoder state.
            out_BPD = h_BPD
            for block in self.blocks[loop_end:]:
                out_BPD = block(out_BPD, ctx_BLD, k_BD)
            predictions.append(self.action_out(modulate(self.final_norm(out_BPD), shift, scale)))
        return torch.stack(predictions)

    def compute_loss(self, batch):
        action_BPA = batch["action"]
        k_B = torch.randint(0, len(self.alpha_bar_K), (action_BPA.shape[0],), device=self.device)
        a_B11 = self.alpha_bar_K[k_B][:, None, None]
        noise_BPA = torch.randn_like(action_BPA)
        x_BPA = a_B11.sqrt() * action_BPA + (1 - a_B11).sqrt() * noise_BPA
        predictions_RBPA = self.denoise_loops(x_BPA, k_B, self.encode(batch))
        losses_R = torch.stack([F.mse_loss(prediction_BPA, noise_BPA) for prediction_BPA in predictions_RBPA])
        loss = losses_R[-1]
        if self.n_loops > 1:
            loss = loss + losses_R[:-1].mean()
        return {"loss": loss,
                **{f"loss_loop_{r}": loss.detach() for r, loss in enumerate(losses_R, 1)}}
