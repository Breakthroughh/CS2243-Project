"""Modal launcher. Humans only: `modal run --detach --env=$CS2243_PROJECT_ENV compute/app.py --name <run> --overrides "..."`.
Comma-separated override values (`train.lr=1e-3,1e-4`) expand into a sweep; one container per config.
With --detach the runs survive closing the terminal; track with `modal app logs <app id>` / WandB."""
from pathlib import Path
import itertools
import os
import subprocess

import modal

ROOT = Path(__file__).parents[1]
LOG = ROOT / "EXPERIMENT_LOG.md"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
PROJECT = os.environ.get("CS2243_PROJECT_ENV", "").strip()
VOLUME_NAME = os.environ.get("CS2243_VOLUME_NAME", "").strip()
PURPOSE = os.environ.get("CS2243_PURPOSE", "pilot")  # onboarding|pilot|baseline|main|ablation|evaluation
if not PROJECT or not VOLUME_NAME:
    raise ValueError("Set CS2243_PROJECT_ENV and CS2243_VOLUME_NAME to the staff-assigned names.")
app = modal.App(f"{PROJECT}-libero", tags={"project": PROJECT, "purpose": PURPOSE})
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libegl1", "libgl1", "libglib2.0-0")
    .uv_pip_install("torch==2.2.2")  # default PyPI wheel = CUDA 12.1
    .uv_pip_install(requirements=[str(ROOT / "compute/requirements.txt")])
    .run_commands("git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
                  f"git -C /opt/LIBERO checkout {LIBERO_COMMIT}")
    .uv_pip_install("/opt/LIBERO", extra_options="--no-deps")  # imports still resolve via PYTHONPATH
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
          "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
          "LIBERO_CONFIG_PATH": "/root/.libero", "PYTHONPATH": "/root:/opt/LIBERO",
          "CS2243_PROJECT_ENV": PROJECT, "CS2243_VOLUME_NAME": VOLUME_NAME,
          "CS2243_PURPOSE": PURPOSE,  # module is re-imported in the container
          # model downloads (t5-base, MiniLM, converted policies, ImageNet weights) cached once on the Volume
          "HF_HOME": "/results/cache/huggingface", "TORCH_HOME": "/results/cache/torch"})
    .add_local_dir(ROOT / "src", "/root/src", ignore=["**/__pycache__"])
)


def check_environment():
    if os.environ.get("MODAL_ENVIRONMENT") != PROJECT:
        raise RuntimeError(f"Wrong Modal Environment; run with --env={PROJECT}")


def git_info():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


@app.function(image=image, timeout=300, max_containers=1)
def tasks(suite: str):
    check_environment()
    from src.rollout import libero_tasks
    lines = [f"{i}: {t['name']} | {t['instruction']}" for i, t in enumerate(libero_tasks(suite))]
    print("\n".join(lines))
    return lines


@app.function(image=image, gpu="L4", cpu=4, memory=16384, volumes={"/results": volume},
              secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")],
              timeout=86400, max_containers=4)  # 24 h (course max): rt1x 50k steps ~14.4 h + 3 x 520-episode evals
def run(cfg: dict):
    check_environment()
    from importlib.metadata import distributions
    from omegaconf import OmegaConf
    from src.run import main as run_main
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    pins = {}
    for d in distributions():  # sys.path order: first hit per name is the one imported (image has 2 site-packages)
        pins.setdefault(d.metadata["Name"].lower().replace("_", "-"), f"{d.metadata['Name']}=={d.version}")
    (run_dir / "freeze.txt").write_text("\n".join(sorted(pins.values())))
    try:
        run_main(OmegaConf.create(cfg))
    finally:
        volume.commit()


@app.local_entrypoint()
def main(name: str, overrides: str = ""):
    if f"`{name}`" not in LOG.read_text():
        raise SystemExit(f"Add a line to EXPERIMENT_LOG.md containing `{name}` and what it tests before launching.")
    from omegaconf import OmegaConf
    from src.config import compose
    branch, commit, dirty = git_info()
    if dirty:
        print("WARNING: uncommitted changes; recorded as git.dirty=True")
    # "a=1,2 b=x" -> [["a=1", "a=2"], ["b=x"]]; list values "[..]" are not split
    axes = [[f"{k}={v}" for v in ([vs] if vs.startswith("[") else vs.split(","))]
            for k, vs in (o.split("=", 1) for o in overrides.split())]
    cfgs = []
    for combo in itertools.product(*axes):
        run_name = "_".join([name] + [o for o, axis in zip(combo, axes) if len(axis) > 1])
        cfg = compose(combo)
        cfg.name, cfg.run_dir = run_name, f"/results/runs/{branch}/{run_name}"
        cfg.git.branch, cfg.git.commit, cfg.git.dirty = branch, commit, dirty
        cfgs.append(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    # .spawn, not .map: map inputs are cancelled if this client dies; spawned calls survive it under --detach
    calls = [run.spawn(cfg) for cfg in cfgs]
    print("Spawned:", *(f"{c['name']}: {call.object_id}" for c, call in zip(cfgs, calls)), sep="\n  ")
    modal.FunctionCall.gather(*calls)
