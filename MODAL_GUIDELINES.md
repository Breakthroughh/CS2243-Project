# CS 2243 Modal requirements and best practices

Source: [course Modal Guide](https://sitanchen.com/cs224/f26/modal_onboarding.html),
read October 19, 2026. This is a paraphrased checklist of that guide, not a complete
assignment specification or a statement that this starter satisfies every item.
The course distinguishes **policy**, **best practices**, and **platform facts**.
Its platform facts and prices were checked September 7, 2026; consult the linked
official documentation and live dashboard for changes. Ask staff about conflicting
billing, permissions, persistence, or credential behavior rather than guessing.

## Course requirements

### Accounts, access, and data

- Use your individual account in the **course Workspace**, with its matching CLI
  profile, and only your team's assigned **restricted project Environment**.
  Keep all project Apps, Secrets, Volumes, and Notebooks there. Do not create extra
  Environments or run sponsored work in a personal Workspace.
- If asked for a personal payment method, unable to see your Environment, or able
  to see another team's restricted Environment, stop and contact staff at
  [cs2243f26staff@gmail.com](mailto:cs2243f26staff@gmail.com). Personal Workspace
  charges cannot be reimbursed. Students are Workspace Members and project
  Contributors; staff manage billing and access.
- Use only freely available, public, open-source data on Modal. Private, licensed,
  confidential, and other restricted datasets are outside the course's approval.
  Discuss a different dataset or arrangement with the instructor before uploading
  data that fails this condition, including scraped data with unsuitable terms.
- Check with staff before adopting other cloud services, APIs, or paid tools;
  Hugging Face and Weights & Biases are explicitly allowed. Discuss external
  compute before combining infrastructures; disclose all compute in the write-up
  and preserve fair, reproducible comparisons.
- Account creation shares name/email with Modal and accepts its terms. Students
  with concerns can request an alternative from the instructor without a project
  or grading penalty.

### Compute limits and first larger run

- **Every Modal Function must explicitly set `max_containers` and `timeout`.**
  Use one container for a single experiment and an intentional parallelism limit
  for a sweep. Choose timeouts slightly above measured pilot durations; the guide
  gives a maximum of 24 hours. Limits apply to CPU helpers as well as GPU jobs.
- Test onboarding on a **single T4**. Use T4 for light work and L4 as the ordinary
  experiment default. Before the team's first use of a GPU other than T4/L4, or a
  run with multiple GPUs per container, post one line on
  [Ed](https://edstem.org/us/courses/104970/) describing the experiment, GPU,
  pilot-measured cost per run, and number of runs. A private staff-visible post is
  sufficient: **no approval or waiting is required**. Later runs in the same family
  need no repeat post. The guide separately flags H200 as normally disallowed
  without a specific need and B200/B300 as normally disallowed; resolve such use
  with staff rather than interpreting the notice rule as unrestricted permission.
- The semester allocation is **$750 per team**, a ceiling rather than a spending
  target. Staff set a budget equal to remaining funds, with an ordinary concurrent
  limit of **4 GPUs and 8 containers**. Ask on Ed for a temporary higher GPU cap.
  `max_containers=1` still permits multiple GPUs if requested within that container.
- Budgets are monthly platform caps, reset by staff to the remaining semester
  allocation. Queued jobs after image build may be waiting on concurrency limits.
  Staff publish cumulative usage; keep your own measured forecast.
- To request reserve funding, give staff the result so far, the question extra
  runs resolve, measured cost/run, run count, and stopping rule. Expect roughly a
  day or two for a reply; an unbounded request to try more ideas is insufficient.
- Avoid region pinning for ordinary course runs. Do not use warm containers,
  unnecessary deployed services, or other cost multipliers for batch experiments.

### Persistent artifacts and retention

- Use the **existing team Volume and exact name supplied by staff**. Ask staff
  about missing names or permissions instead of creating untracked Volumes.
  Only paths below its mount persist; ordinary `/root` and `/tmp` files disappear.
  This repo mounts the Volume at `/results`; the guide's example uses `/mnt/project`.
- Long-running jobs must checkpoint and support resumption. Retain the latest
  fully resumable checkpoint, best checkpoint selected by a predetermined
  validation rule, final quantitative outputs, representative samples, configs,
  logs, and inputs underlying reported figures.
- Remove or archive redundant failed-sweep checkpoints, duplicate caches/model
  downloads, unused raw sample dumps, and cheap-to-recreate temporary artifacts.
  Mention expected storage above approximately **100 GB** on Ed. The course-wide
  free storage allowance is shared, not an allowance for every team.

### Credentials, naming, and reporting

- Store external-service tokens in Modal Secrets within the project Environment;
  attach only to Functions/Notebooks that need them and read injected environment
  variables without printing them. Use project-scoped names and least privilege.
- Never share Modal logins or personal Modal API tokens. Never put credentials in
  Git, notebook cells, configs, tracker fields, logs, screenshots, or returned
  values. Rotate suspected exposed credentials immediately. Inspect complete Git
  history for credentials before publishing a repository.
- A shared project service credential is acceptable only when the provider's
  terms and security model allow it. Do not casually share powerful personal tokens.
- Name Apps with the project name and experiment purpose; include `project` and
  `purpose` tags. Suggested purposes: `onboarding`, `pilot`, `baseline`, `main`,
  `ablation`, and `evaluation`.
- Choose final-comparison seeds in advance and report every completed run,
  including failures. Do not select and report only favorable seeds.
- Use Functions for long training, large evaluations/generation, sweeps, resumable
  work, and final reported artifacts. Do not disable Notebook idle shutdown to
  keep long jobs alive. Do not enable public/unlisted course Notebook sharing
  without approval, especially with Secrets, private data, or unpublished results.

## Onboarding checklist

- [ ] Accept the invitation; verify the course Workspace and restricted team access.
- [ ] Create/activate a local Python environment, install Modal, and authenticate.
- [ ] Identify and activate the course CLI profile.
- [ ] Identify/set the project Environment; verify the resulting CLI configuration.
- [ ] Run the committed `modal_test.py` with an explicit Environment and confirm
      the returned Environment, CUDA computation, T4 identity, and successful exit.
- [ ] Confirm the staff-provided Volume name and mount path.
- [ ] Read the allocation, function limits, and checkpoint requirements before research runs.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
modal setup
modal profile list
modal profile activate COURSE_PROFILE
modal profile current
modal environment list
modal config set-environment PROJECT_ENV
modal config show

# Replace both placeholders with the exact staff-assigned names.
export CS2243_PROJECT_ENV=PROJECT_ENV
export CS2243_VOLUME_NAME=YOUR_VOLUME_NAME
modal run --env="$CS2243_PROJECT_ENV" modal_test.py
```

`modal setup` also works as `python -m modal setup`. Each teammate authenticates
individually. `COURSE_PROFILE`, `PROJECT_ENV`, and `YOUR_VOLUME_NAME` are placeholders,
not defaults to use literally. If onboarding fails, send staff the complete
terminal output and App dashboard link, with no tokens.

## Suggested repository organization

The guide calls this layout a **suggestion**, not a grading requirement:

```text
project-repository/
├── README.md               # research question, setup, exact run commands
├── pyproject.toml          # or another dependency manifest
├── requirements.lock       # exact tested dependency versions, once validated
├── modal_app.py            # Images, Functions, Volumes, Secrets
├── src/                    # research/model/experiment implementation
├── configs/                # immutable experiment configurations
├── scripts/                # evaluation and aggregation entrypoints
├── tests/                  # fast local correctness checks
├── results/                # small final tables/plots; no large checkpoints
└── .gitignore              # credentials, data, caches, environment files
```

This repo additionally keeps `modal_test.py`, `requirements-sim.txt`, and this
document. The simulator lives in `src/sim/`; `modal_app.py` includes `src` and
`configs` in the remote Image. Keep substantial research logic in the team Git
repository, reviewed through branches/PRs. Notebooks should not diverge into a
second authoritative implementation. Multi-file Apps can use module mode
(`modal run -m src.app`) when the App is actually in that module; this starter
keeps its App at the top level and uses `modal run modal_app.py`.

## Best practices for experiments

### Environment and configuration

- Define the Image in code, with stable dependency layers first and frequently
  changing source/configuration layers last to reuse build caches.
- After the baseline works, pin exact tested dependencies, Python version, and
  upstream revisions. Record dependency changes with the config and Git commit;
  avoid upgrades midway through a main comparison.
- **Current limitation:** `requirements.lock` is explicitly a placeholder.
  `requirements-sim.txt` contains candidate direct pins; `modal_app.py` pins torch
  and LIBERO. No successful remote build has validated a complete dependency lock.
  After the first successful smoke run, preserve its `pip-freeze.txt`, record the
  Image/Python/LIBERO revision, replace the placeholder with tested versions, and
  separately freeze the local Modal SDK. Review nonportable editable package paths.
- Save complete, immutable configs under `configs/`. Record the exact launch
  command, config, and source revision; commit small reported artifacts in `results/`.

### Scale from measurements

1. Run local correctness checks and tiny synthetic cases.
2. Try a short remote batch or a few optimization steps.
3. Pilot long enough to measure throughput, memory, checkpoint size, evaluation
   cost, and checkpoint/resume behavior.
4. Screen configurations cheaply with one seed.
5. Lock methods, baselines, metrics, and seeds for the main comparison.
6. Change one causal component per targeted ablation.
7. Replicate across the predetermined seeds once implementation and metrics are stable.

Use throughput and cost per unit of work to choose hardware. A faster GPU may be
cheaper overall for compute-bound training; the guide's H100-versus-L4 example is
not a guarantee for this simulator. Right-size CPU/RAM to measurements. Inspect
GPU memory, utilization/power, CPU/RAM, throughput, metric plausibility, and
checkpoint health before increasing hardware or scale. Investigate data loading,
preprocessing, synchronization, tiny batches, and logging when utilization is low.
For OOMs, examine batch size, accumulation, and activation checkpointing first.

### Checkpoint correctly

Save model, optimizer, scheduler, step/epoch, seeds and relevant RNG/sampler state,
mixed-precision scaler if used, complete config, base-model revision, dataset
version, Git commit, dependency fingerprint, and sanity-check metrics. Test resume.
Write to a temporary filename and atomically rename a completed checkpoint. Use
a separate directory for each run. Call `volume.commit()` before publishing writes
to a reader or returning when needed. Existing readers must close open Volume
files and call `volume.reload()` to see another container's updates.

Detached runs survive laptop disconnects but still require checkpoints. The guide
states GPU Functions cannot use the paid non-preemptible option. Optional Beta
Notebooks suit short interactive analysis; retain the default ten-minute idle
shutdown, persist under an attached Volume, and keep source in Git.

### Keep a shared run and compute log

Use `results/run-log.csv` as a starting schema. For each run record:

- Unique run ID, scientific purpose/hypothesis, and next decision.
- Git commit and clean/dirty state, dependency lock/fingerprint, Image definition.
- Dataset version/split/preprocessing and base-model identifier/revision.
- Complete config, seed, precision, batch/accumulation, schedule, checkpoint source.
- GPU type/count, physical CPU cores, RAM, timeout, and container cap.
- Pilot-based forecast, actual cost/runtime, status/failure, metrics, artifact paths.

Maintain a compute plan listing experiment, hardware, hours/run, configuration
count, seed count, estimated total cost, and stopping rule. Count all containers,
GPUs per container, evaluation, failures, and retries. Coordinate team launches to
avoid duplicate sweeps. Before an expensive run, check profile/Environment,
explicit `--env`, Function limits, working checkpoint/resume, and team coordination.

## Cost reference and useful commands

The following are **historical rates from the guide**, not newly verified prices.
Verify [current pricing](https://modal.com/pricing) before using them for a budget.

| Standard Function resource | Guide rate per hour |
| --- | ---: |
| T4 | $0.5904 |
| L4 | $0.7992 |
| A10 | $1.1016 |
| L40S | $1.9512 |
| A100 40 GB / 80 GB | $2.0988 / $2.4984 |
| RTX PRO 6000 | $3.0312 |
| H100 / H200 | $3.9492 / $4.5396 |
| B200 / B300 | $6.2496 / $7.0992 |
| Physical CPU core (2 vCPU) | $0.04716 |
| Memory per GiB | $0.007992 |

Estimate runtime hours × concurrent runs × (GPU count × GPU hourly rate + CPU
cores × CPU hourly rate + GiB × memory hourly rate). CPU/memory billing uses the
larger of requested and actual use. The guide lists Notebook/Sandbox CPU at
$0.141912/core-hour and memory at $0.024012/GiB-hour, with the same GPU rates.
Region multipliers are 1.15× broad and 1.75× narrow. Storage includes a shared
1 TiB-month, then $0.09/GiB-month. Its egress notice specifies billing from October
1, 2026, a shared Team-plan 10 TiB/cycle allowance, then $0.04/GiB; Volume traffic
is excluded. First egress-inclusive bill: November 1, 2026. Storage is outside
the team's compute budget; account for other billable overhead when forecasting.

```sh
modal profile current
modal config show
modal volume list --env="$CS2243_PROJECT_ENV"
modal volume ls --env="$CS2243_PROJECT_ENV" "$CS2243_VOLUME_NAME" /
modal volume get --env="$CS2243_PROJECT_ENV" "$CS2243_VOLUME_NAME" /path/in/volume ./local-output
modal run --env="$CS2243_PROJECT_ENV" modal_app.py --mode list
modal run --env="$CS2243_PROJECT_ENV" --detach modal_app.py
modal app logs --env="$CS2243_PROJECT_ENV" APP_ID
modal app stop --env="$CS2243_PROJECT_ENV" APP_ID
```

Use `modal run` for one-off work and `--detach` if it must survive a disconnect;
reserve `modal deploy` for genuinely persistent services. If infrastructure seems
broken, inspect logs and [Modal status](https://status.modal.com/). For a discrepancy,
send staff the guide section, conflicting sentence/command, official source or
live output, and observation date.

Further official references: [project structure](https://modal.com/docs/guide/project-structure),
[Images](https://modal.com/docs/guide/images), [Environments](https://modal.com/docs/guide/environments),
[scaling](https://modal.com/docs/guide/scale), [timeouts](https://modal.com/docs/guide/timeouts),
[Volumes](https://modal.com/docs/guide/volumes), [Secrets](https://modal.com/docs/guide/secrets),
[preemption](https://modal.com/docs/guide/preemption), and [Notebooks](https://modal.com/docs/guide/notebooks).
