# Experiment log

<!-- HUMANS ONLY; agents must not edit this file. The launcher refuses any run whose `name` isn't in this file.

## <Experiment title> (branch `exp/<name>`)
<Two sentences: the question, and the hypothesis being tested.>
- YYYY-MM-DD HH:MM EST `<run-name>`: what this ablation tests. Changes: ... Expected: ... Result: ...

Failed arms: branch, code changes, why it failed. -->

- 2026-10-21, 19:21 EST `e2e-smoke`: hello world, does this work lol
    - Changes: set up infra
    - Expected: this shit works
    - Results: yay

## vla-baselines
- 2026-11-03, 15:01 EST `baseline`: baseline standard policy from a pretrained VLA-style model as well as training infrastructure to define a baseline
    - Changes: add new Torch implementations of accepted architectures (Octo 27/93M + RT-1-X 35M) that are generalist robot policies with transformer backbones as well as training loop based on the dit-baseline similar training loop; train on the same LIBERO-Spatial initial task set with IDENTICAL training settings to attempt to reproduce, within noise, the performance of these policies on tasks as reported by their original papers
    - Expected: slightly different results but within noise of the original papers' performances; baselines should also operate within noise of the diffusion policy-style baselines that we established in WandB and on the other branch (roughly 73% performance at 50k steps final eval)
    - Results: 
- 2026-11-04, 17:20 EST `octo-smoke`: Octo-small random weights port baseline; prepare the model and the training loop / setup given from the OpenVLA paper
    - Changes: set up Octo on PyTorch
    - Expected: roughly on policy with the Octo paper
- 2026-11-04, 17:44 EST `octo-small-50k-b64`: Octo-small training baseline
    - Changes: N/A
    - Expected: 