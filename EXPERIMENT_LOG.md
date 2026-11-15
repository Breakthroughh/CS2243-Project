# Experiment log

<!-- HUMANS ONLY; agents must not edit this file. The launcher refuses any run whose `name` isn't in this file.

## <Experiment title> (branch `exp/<name>`)
<Two sentences: the question, and the hypothesis being tested.>
- YYYY-MM-DD HH:MM EST `<run-name>`: what this ablation tests. Changes: ... Expected: ... Result: ...

Failed arms: branch, code changes, why it failed. -->

- 2026-10-21, 19:21 EST `e2e-smoke`: hello world, does this work lol
    - Changes: set up infra
    - Expected: this shit works
    - Results: 
- 2026-11-03: `dit-pilot`
    - Changes: running a test first one
    - Expected: i expect it to be ok
    - Results: its ass
    - running da pilot fr
    - Results: yay
- 2026-11-03, 15:01 EST `baseline`: baseline standard policy from a pretrained model as well as training infrastructure to define a baseline
    - Changes: add new diffusion policy as a subclass + training loop to train; set up task split 
    - Expected: 
    - Results: 
- 2026-11-03, 8:49pm `baseline-dit`: longer dit
    - Changes: increase training length 
    - Expected: 50% success on libero
    - Results: test score 87%

- 2026-11-08 Note:
    - Want to modify Advikar's baseline-dit to loop as well (an additional inner loop using shared-weight layers for refinement, not just denoising steps), referencing Looped Diffusion Transformer paper https://arxiv.org/pdf/2609.40305, which uses 1. deep supervision so intermediate states become useful predictions themselves, and 2. self modulating attention to fix redundant attention updates when looping.
    - Will first implement looping on 2/6 blocks (2 pre, 2 looped, 2 post) for ablation test.

- 2026-11-08, 5:39pm `simple-looped-dit-smoke`: Smoke test for below
    - Changes: See below
    - Expected: Working
    - Results: 
- 2026-11-08, 5:50pm `simple-looped-dit`: Naively change baseline to have middle 2 blocks loop twice
    - Changes: Implement looping on 2/6 blocks (2 pre, 2 looped, 2 post)
    - Expected: Same or slight degradation in performance compared to baseline-dit
    - Results: 79.4% on LIBERO Spatial as expected

- 2026-11-08, 2:55pm `deepsup-looped-dit`: Add deep supervision to `simple-looped-dit`, still 2 loops
    - Changes: Add deep supervision as in Looped DiT paper. Here loss fxn is just the sum L1 + L2.
    - Expected: Improvement on `simple-looped-dit`
    - Results: 84.8% on LIBERO Spatial, ok but not exceeding baseline

- 2026-11-10, 11:57am `deepsup-xsa-looped-dit`: Add XSA to `deepsup-looped-dit`, still 2 loops
    - Changes: Add XSA as in Looped DiT paper. Here loss fxn is just the sum L1 + L2.
    - Expected: Slight improvement on `deepsup-looped-dit`
    - Results: 88% on LIBERO Spatial, ok

- 2026-11-15, 3:35pm `l4-deepsup-xsa-looped-dit`: Above, but loop 4 times
    - Changes: 4 loops instead of 2
    - Expected: Slight improvement on `deepsup-xsa-looped-dit`
    - Results: 