"""Run LIBERO on Modal without selecting or downloading any policy model."""
from pathlib import Path
import os
import modal

ROOT = Path(__file__).parent
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
PROJECT = os.environ.get("CS2243_PROJECT_ENV", "").strip()
VOLUME_NAME = os.environ.get("CS2243_VOLUME_NAME", "").strip()
if not PROJECT or not VOLUME_NAME:
    raise ValueError("Set CS2243_PROJECT_ENV and CS2243_VOLUME_NAME to the staff-assigned names.")
app = modal.App(f"{PROJECT}-libero-smoke", tags={"project": PROJECT, "purpose": "pilot"})
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libegl1", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.2.2", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "requirements-sim.txt"))
    .run_commands("git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
                  f"git -C /opt/LIBERO checkout {LIBERO_COMMIT}",
                  "pip install --no-deps -e /opt/LIBERO")
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
          "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
          "LIBERO_CONFIG_PATH": "/root/.libero", "PYTHONPATH": "/root:/opt/LIBERO"})
    .add_local_python_source("src")
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs")
)

def check_environment():
    if os.environ.get("MODAL_ENVIRONMENT") != PROJECT:
        raise RuntimeError(f"Wrong Modal Environment; run with --env={PROJECT}")


@app.function(image=image, timeout=300, max_containers=1)
def tasks(suite: str):
    check_environment()
    from src.sim.rollout import list_tasks
    return list_tasks(suite)

@app.function(image=image, gpu="L4", cpu=2, memory=8192,
              volumes={"/results": volume}, timeout=600, max_containers=1)
def smoke(suite: str, task_id: int, init_state_id: int, steps: int, seed: int):
    import json
    import subprocess
    import sys
    import traceback
    import uuid
    check_environment()
    from src.sim.rollout import rollout
    directory = Path("/results") / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    (directory / "environment.json").write_text(json.dumps({"libero_commit": LIBERO_COMMIT}))
    (directory / "pip-freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    try:
        summary = rollout(directory, suite, task_id, init_state_id, steps, seed)
        # Return small artifacts to laptop automatically, as well as keeping Volume copies.
        files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        return {"run_id": directory.name, "summary": summary, "files": files}
    except Exception:
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        volume.commit()
        print(f"Persistent outputs: {directory}")

@app.local_entrypoint()
def main(mode: str = "smoke", suite: str = "libero_spatial", task_id: int = 0,
         init_state_id: int = 0, steps: int = 60, seed: int = 0):
    import json
    if suite not in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        raise ValueError("Unsupported suite")
    if mode == "list":
        print(json.dumps(tasks.remote(suite), indent=2))
        return
    if mode != "smoke" or not 1 <= steps <= 500:
        raise ValueError("Use mode list or smoke, and steps 1..500")
    result = smoke.remote(suite, task_id, init_state_id, steps, seed)
    output = ROOT / "outputs" / result["run_id"]
    output.mkdir(parents=True)
    for name, data in result["files"].items():
        (output / name).write_bytes(data)
    print(json.dumps(result["summary"], indent=2))
    print(f"Local video: {output / 'rollout.mp4'}")
