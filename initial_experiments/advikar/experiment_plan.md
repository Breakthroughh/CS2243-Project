# Experiment plan: looped transformer diffusion heads

Goal: build a looped transformer diffusion head, learn where to loop, and transfer it to a VLA.
Order: start on a small diffusion policy (cheap, fast iteration on LIBERO-spatial), then move to the VLA once looping
is shown to matter.

Current repo state: only `NoOpPolicy` + training/eval scaffolding exist.

## Phase 0: Non-looped baseline
1. **DiT-style diffusion policy**: ResNet-18 encoder (agentview + wrist) + state + lang embedding -> transformer
   denoiser (D blocks) over the action chunk `[B,P,A]`, P=16. New `configs/policy/dit.yaml` on the `Policy` interface.
2. **Pilot**: a few hundred steps + one eval pass to measure $/run (training *and* eval: 50 inits x 10 tasks = 500
   episodes). Sets the ablation budget (L4 ~ $0.80/hr, $750 total).
3. **Full baseline run**: the reference number for everything after.

## Phase 1: Fixed looping (does looping help at all?)
4. **Looped head**: L shared blocks x K loops at matched FLOPs (e.g. D=6 vs L=2, K=3). Intermediate-loop supervision
   (decode every loop, loss on each), as in Looped-DiT.
5. **Random-K training + inference-time K sweep**: train with K ~ U{1..6}; eval one checkpoint at K=1..6, and compare
   extra loops vs extra denoising steps at equal inference compute.
   - Key go/no-go check: if success doesn't change with K, or no states benefit from more loops, there is nothing for
     a scheduler to learn.

## Phase 2: Learn where to loop
6. **Non-learned baseline**: halt when the latent stops changing (RD-VLA delta threshold). Learned schedulers must beat it.
7. **Layer level (K_i)**: halting head trained with BC loss + compute penalty (PonderNet-style). Per denoising step first,
   then per layer.
8. **Token level**: sigmoid gate before the FFN choosing which action tokens get fresh FFN compute vs cached outputs.
   Ablate layer level, token level, then both (<= 2 changes per ablation).
9. **Optional: RL fine-tune of the scheduler** (PPO on success minus compute, like D3P).
   - BC is a warm start: it optimizes action-matching (denoising error), not task success, so a BC-trained scheduler
     learns where loops reduce denoising error, which isn't necessarily where they increase success. RL closes that gap
     but needs sim rollouts; only do it if Phase 2 shows a gap worth closing.

## Phase 3: Move to the VLA
10. **Frozen VLM backbone** (RD-VLA's Qwen2.5-0.5B-based model; verify which weights are released) + best looped head
    from Phase 2.
11. **Precompute VLM features** for all demo frames once, cache in `/results/datasets/`. Head training then costs about
    as much as Phase 1; only eval needs the live VLM (fits on L4).
12. **Repeat steps 5-8 on the VLA**; compare to RD-VLA's non-diffusion loop.

## First experiment: `exp/looped-dit-head`
- Runs: `dit-baseline` (D=6), `looped-dit` (L=2, random K up to 6, intermediate supervision), then the K-sweep eval on
  `looped-dit`.
- Code: `src/diffusion.py` (denoiser + loop option), two policy yamls, tests first (shapes, loss decreases on a tiny
  batch, `predict_action` -> `[B,P,A]`, K changes compute but not output shape).
- Hypotheses, expectations, and results go in `EXPERIMENT_LOG.md`, human-written, before each run.
