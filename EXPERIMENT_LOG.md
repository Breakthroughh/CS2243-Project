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
- 2026-11-03, 15:01 EST `baseline`: baseline standard policy from a pretrained model as well as training infrastructure to define a baseline
    - Changes: add new diffusion policy as a subclass + training loop to train; set up task split 
    - Expected: 
    - Results: 