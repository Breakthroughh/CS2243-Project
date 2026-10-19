# LIBERO simulation starter

Shared simulation infrastructure for a robotics research project. No policy model
has been chosen. No model weights or demonstration datasets are downloaded.

**Status:** Python syntax checked; not yet built or run on Modal. The first team
milestone is verifying installation, offscreen rendering, and saved observations.
This is a proposed compatible dependency set, not a fully locked environment.

## What runs where?

Your laptop edits code and launches jobs. Modal runs LIBERO / robosuite / MuJoCo
on Linux. The L4 GPU is used for camera rendering, not model inference.
PyTorch is CPU-only here because LIBERO uses it to load initial states.
A future GPU policy needs a CUDA-enabled PyTorch environment of its own.

## Quick start (Mac, Linux, or Windows with an appropriate Python shell)

Use Python 3.10 or newer locally. On Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install modal
modal setup
modal run modal_app.py --mode list
modal run modal_app.py --mode smoke --task-id 0 --steps 60
```

The first run builds a reusable cloud image. A smoke test runs in one container
with a ten-minute timeout. It sends zero pose commands and an open-gripper command;
the robot should mostly stay still. This is NOT expected to solve the task.

Open the printed `outputs/<run-id>/rollout.mp4` on your laptop. That folder also
contains two camera snapshots, `last_observation.npz`, `summary.json`, upstream
commit information, and `pip-freeze.txt`. Copies persist in the Modal Volume
`libero-starter-results`. Failed runs save `error.txt` there when possible.

Try another task or initial state:

```bash
modal run modal_app.py --task-id 1 --init-state-id 2
modal run modal_app.py --suite libero_goal --task-id 0
```

## What is in this repo?

- `modal_app.py`: cloud environment, task listing, remote launch, result download.
- `sim/rollout.py`: configuration, environment creation, initial state, stepping,
  success check, observations and video. This is independent of a policy model.
- `requirements-sim.txt`: selected simulator dependencies.
- `.gitignore`: excludes generated outputs, weights, datasets, and credentials.

The official LIBERO repository is installed at a pinned commit inside the image;
we do not copy or modify its source in this repository.

## Sharing on GitHub

Create an EMPTY private GitHub repository named `libero-starter` (no README).
Inside this downloaded folder, run:

```bash
git init -b main
git add .
git commit -m "Add policy-independent LIBERO simulation starter"
git remote add origin https://github.com/YOUR_USERNAME/libero-starter.git
git push -u origin main
```

Replace YOUR_USERNAME. Add teammates in repository Settings > Collaborators.
Each teammate clones the repository and follows Quick start. Each runs `modal
setup` with their own account. To use the same credits/Volume, they need access
to the same Modal workspace; GitHub access alone does not grant that access.
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
