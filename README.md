# LIBERO simulation starter

Shared simulation infrastructure for a robotics research project. No policy model
has been chosen. No model weights or demonstration datasets are downloaded.

**Research question:** not yet selected. Define it with the team before choosing
a policy or launching the main comparison.

Read [MODAL_GUIDELINES.md](MODAL_GUIDELINES.md) for the course requirements,
onboarding checklist, compute/storage policies, and experiment best practices.

**Status:** Python syntax checked; not yet built or run on Modal. The first team
milestone is verifying installation, offscreen rendering, and saved observations.
This is a proposed compatible dependency set, not a fully locked environment.
`requirements.lock` is a documented placeholder to fill after the first successful
remote run; do not install it as though it were a validated lockfile.

## What runs where?

Your laptop edits code and launches jobs. Modal runs LIBERO / robosuite / MuJoCo
on Linux. The L4 GPU is used for camera rendering, not model inference.
PyTorch is CPU-only here because LIBERO uses it to load initial states.
A future GPU policy needs a CUDA-enabled PyTorch environment of its own.

## Quick start (Mac, Linux, or Windows with an appropriate Python shell)

Install [uv](https://docs.astral.sh/uv/). It pins Python 3.10 (`.python-version`) to match the Modal image:

```bash
uv sync
source .venv/bin/activate   # or prefix commands with `uv run`
modal setup
modal profile list
modal profile activate COURSE_PROFILE
modal profile current
modal environment list
modal config set-environment PROJECT_ENV
modal config show

# Replace these with the exact Environment and existing Volume supplied by staff.
export CS2243_PROJECT_ENV=PROJECT_ENV
export CS2243_VOLUME_NAME=YOUR_VOLUME_NAME

# Complete course onboarding first: one T4, one container, five-minute limit.
modal run --env="$CS2243_PROJECT_ENV" modal_test.py

# Then verify the simulator on L4.
modal run --env="$CS2243_PROJECT_ENV" modal_app.py --mode list
modal run --env="$CS2243_PROJECT_ENV" modal_app.py --mode smoke --task-id 0 --steps 60
```

Run from the repository root. `COURSE_PROFILE`, `PROJECT_ENV`, and
`YOUR_VOLUME_NAME` are placeholders: discover the first two with the list commands
and obtain the Volume name from staff. Export the two `CS2243_` variables in each
new shell. The launcher refuses missing names and checks the remote Environment;
it uses the existing Volume without creating one. No personal payment method is
needed. Stop and contact staff if billing or restricted access looks wrong.

The first run builds a reusable cloud image. A smoke test runs in one container
with a ten-minute timeout. It sends zero pose commands and an open-gripper command;
the robot should mostly stay still. This is NOT expected to solve the task.

Open the printed `outputs/<run-id>/rollout.mp4` on your laptop. That folder also
contains two camera snapshots, `last_observation.npz`, `summary.json`, upstream
commit information, and `pip-freeze.txt`. Copies persist in the Modal Volume
named by `CS2243_VOLUME_NAME`, mounted at `/results`. Failed runs save `error.txt`
there when possible. Local downloads remain in ignored `outputs/`; only small
reported tables and plots belong in `results/` in Git.

Try another task or initial state:

```bash
modal run --env="$CS2243_PROJECT_ENV" modal_app.py --task-id 1 --init-state-id 2
modal run --env="$CS2243_PROJECT_ENV" modal_app.py --suite libero_goal --task-id 0
```

## What is in this repo?

```text
.
├── README.md
├── MODAL_GUIDELINES.md       # course policies and best practices
├── pyproject.toml           # local package and Modal dependency
├── requirements.lock        # pending exact tested remote dependency lock
├── requirements-sim.txt     # candidate direct simulator dependency pins
├── modal_app.py             # Image, Functions, existing Volume, local launch
├── modal_test.py            # course T4 onboarding check
├── src/
│   └── sim/rollout.py        # policy-independent simulator implementation
├── configs/                 # immutable experiment configurations
├── scripts/                 # future evaluation and aggregation entrypoints
├── tests/                   # fast checks without cloud dependencies
├── results/                 # small reported tables/plots and shared run log
└── .gitignore
```

`src/sim/rollout.py` handles environment creation, initial states, stepping,
success checks, observations, and video. The Image includes the `src` package and
copies `configs/` to `/root/configs`. The current CLI takes arguments directly;
config-driven research and full policy evaluation are still to be implemented.

Run the local checks with `uv run pytest -q`. After a
successful remote smoke run, use the saved `pip-freeze.txt` and image/upstream
metadata to produce the tested lock described in `requirements.lock`.

The official LIBERO repository is installed at a pinned commit inside the image;
we do not copy or modify its source in this repository.

## Sharing on GitHub

Use the team's private GitHub repository and add teammates as collaborators.
Each teammate clones the repository and follows Quick start. Each runs `modal
setup` with their own account. To use the same credits/Volume, they need access
to the course Workspace and the same restricted project Environment; GitHub
access alone does not grant that access. Use only the staff-provided team Volume.
Never commit or share Modal authentication tokens. Work in branches and use PRs.

## First team tasks

1. Infrastructure: run the smoke test, fix dependency/rendering issues, document
   the first successful run and dependency versions.
2. Environment exploration: inspect task names, images, observation keys, and
   initial-state IDs. Select a small initial evaluation set.
3. Policy exploration: choose a baseline and design an adapter that consumes an
   observation plus instruction and returns actions in the simulator convention.

Only after selecting a policy: add model-specific image preprocessing, action
normalization, chunk execution, and inference timing. Do not feed arbitrary model
outputs directly into env.step. The current environment uses OSC_POSE control:
7 controls (three translation, three rotation, and one gripper command).

The smoke test is NOT a full evaluator. A real evaluator must define episode
limits, settling steps, task/initial-state manifests, model reset semantics,
success aggregation, latency timing, and checkpoint provenance.

## Troubleshooting

- Installation error: copy the first substantive error from the Modal build log.
  Dependencies are older and may require an additional transitive-version pin.
- EGL/rendering error: verify the L4 allocation and EGL environment variables.
- No task success: expected for the no-op policy.
- Laptop import errors for robosuite: use `modal run`, not `python modal_app.py`;
  simulator dependencies are intentionally installed only in the cloud.
- Keep videos/observations out of Git; use Modal storage or an agreed results store.

## References

- https://github.com/Lifelong-Robot-Learning/LIBERO
- https://modal.com/docs/guide/getting-started
- https://modal.com/docs/guide/images


God help us.
