# Diffusion robot policies on LIBERO

Diffusion-policy research on LIBERO, run on Modal, tracked in WandB, checkpoints on HF.
**Read [AGENTS.md](AGENTS.md) first** (protocol, workflow, metrics, code idioms). Course rules: [MODAL_GUIDELINES.md](MODAL_GUIDELINES.md).

## Setup (once)

```bash
uv sync                                    # local .venv (Python 3.10, from uv.lock)
uv run modal setup && uv run modal profile activate <COURSE_PROFILE>
source .env                                # exports CS2243_PROJECT_ENV, CS2243_VOLUME_NAME (staff-assigned)
uv run modal secret create --env="$CS2243_PROJECT_ENV" wandb WANDB_API_KEY=<key>
uv run modal secret create --env="$CS2243_PROJECT_ENV" huggingface HF_TOKEN=<write token for jackfanml>
uv run modal run --env="$CS2243_PROJECT_ENV" compute/onboarding.py   # course T4 check
uv run pytest -q                                                      # local, mocked, CPU-only
```

## Run (humans only)

1. Work in a worktree: `git worktree add .claude/worktrees/<name> -b exp/<name>`.
2. Add a line containing `` `<run>` `` to [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) saying what it tests. The launcher refuses unlogged names.
3. Launch:
   ```bash
   uv run modal run --env="$CS2243_PROJECT_ENV" compute/app.py --name <run> --overrides "<hydra overrides>"
   ```
   Comma values sweep (`train.lr=1e-3,1e-4`). Add `hf.namespace=jackfanml` to push the final policy.

Examples:
- Smoke: `--name e2e-smoke --overrides "train.epochs=1 train.max_steps=1 eval.n_train=1 eval.n_test=1"`
- No-op baseline: `--name noop-baseline --overrides "eval.n_test=50"`

## Where things go

| What | Where |
| --- | --- |
| Run outputs (`config.yaml`, `freeze.txt`, `metrics.jsonl`, `latest.pt`, `videos/`, `wandb/`) | Volume `$CS2243_VOLUME_NAME`: `/runs/<branch>/<run>/` |
| LIBERO datasets (HF `yifengzhu-hf/LIBERO-datasets`, downloaded once) | Volume: `/datasets/LIBERO/` |
| Metrics + videos | WandB `jackfandev-harvard-university/cs2243-diffusion-policy`, group = branch |
| Final policy (opt-in) | HF private repo `jackfanml/<branch>-<run>` |
| Reported tables/plots | `results/` in Git |

Browse or fetch outputs: `uv run modal volume ls|get --env="$CS2243_PROJECT_ENV" "$CS2243_VOLUME_NAME" /runs/...`.
Delete smoke or failed runs once you've processed them (Volume dir, WandB run, HF repo).

## Code

```text
compute/app.py         Modal image + launcher (log gate, sweeps, per-branch dirs); compute/requirements.txt = remote pins
configs/               Hydra groups: policy/ data/ eval/ train/ (data + eval suite gates)
src/run.py             data -> policy -> cfg.train.loop -> final eval -> HF push
src/policies/          baseline.py = Policy interface (HF PyTorchModelHubMixin) + NoOpPolicy; one file per policy
src/training/          baseline.py = loop scaffolding; torch logic in make_optimizer / update; one file per loop
src/{data,rollout,logger,checkpoint}.py
tests/                 mocked CPU tests; test_train.py = training-loop contract
requirements.lock      tested remote lock (from e2e-smoke); uv.lock = local lock
```

New policy: `src/policies/<x>.py` + `configs/policy/<x>.yaml` (`_target_`), select with `policy=<x>`.
New training loop: `src/training/<x>.py:train` + `configs/train/<x>.yaml` (`loop:`), select with `train=<x>`.

God help us.
