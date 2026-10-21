"""Course onboarding check: one T4, one container, five minutes maximum."""
import os
import platform

import modal

PROJECT = os.environ.get("CS2243_PROJECT_ENV", "").strip()
if not PROJECT:
    raise ValueError("Set CS2243_PROJECT_ENV to the staff-assigned Environment name.")

app = modal.App(f"{PROJECT}-test", tags={"project": PROJECT, "purpose": "onboarding"})
# Candidate versions shared with the starter; the research lock remains pending.
image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "torch==2.2.2", "numpy==1.26.4"
).env({"CS2243_PROJECT_ENV": PROJECT})  # module is re-imported in the container


@app.function(image=image, gpu="T4", cpu=0.5, memory=4096,
              timeout=300, max_containers=1)
def inspect_remote_runtime():
    import torch

    if os.environ.get("MODAL_ENVIRONMENT") != PROJECT:
        raise RuntimeError(f"Wrong Modal Environment; run with --env={PROJECT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    value = torch.ones((2, 2), device="cuda")
    if (value @ value).sum().item() != 8.0:
        raise RuntimeError("GPU calculation failed")
    return {"modal_environment": os.environ["MODAL_ENVIRONMENT"],
            "python": platform.python_version(), "cuda_available": True,
            "gpu": torch.cuda.get_device_name(0)}


@app.local_entrypoint()
def main():
    for key, value in inspect_remote_runtime.remote().items():
        print(f"{key}: {value}")
