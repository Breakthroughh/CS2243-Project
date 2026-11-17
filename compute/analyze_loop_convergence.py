"""Human-launched, checkpoint-only analysis for the existing 2/4/6-loop XSA runs."""
import modal

from compute.app import PROJECT, ROOT, check_environment, git_info, image, volume

app = modal.App(f"{PROJECT}-loop-convergence", tags={"project": PROJECT, "purpose": "evaluation"})
analysis_image = (
    image.add_local_dir(ROOT / "scripts/tmp_loop_convergence", "/root/scripts/tmp_loop_convergence",
                        ignore=["**/__pycache__", "test_*"])
    # Modal re-imports this module in the runner; its compute.app dependency must exist there.
    .add_local_dir(ROOT / "compute", "/root/compute", ignore=["**/__pycache__"])
)
RUNS = [(2, "deepsup-xsa-looped-dit"), (4, "l4-deepsup-xsa-looped-dit"),
        (6, "l6-deepsup-xsa-looped-dit")]


@app.function(image=analysis_image, gpu="T4", cpu=2, memory=8192, volumes={"/results": volume},
              timeout=1800, max_containers=1)
def analyze(output_dir: str, samples_per_task: int, batch_size: int, seed: int, revision: dict):
    check_environment()
    import json
    from pathlib import Path
    from scripts.tmp_loop_convergence.analysis import analyze as analyze_checkpoints
    specs = [dict(loops=loops, run=name, path=f"/results/runs/exp/dit-looped/{name}") for loops, name in RUNS]
    try:
        analyze_checkpoints(specs, output_dir, samples_per_task=samples_per_task, batch_size=batch_size, seed=seed)
        (Path(output_dir) / "analysis_revision.json").write_text(json.dumps(revision, indent=2))
    finally:
        volume.commit()


@app.local_entrypoint()
def main(name: str, samples_per_task: int = 16, batch_size: int = 16, seed: int = 0):
    if f"`{name}`" not in (ROOT / "EXPERIMENT_LOG.md").read_text():
        raise SystemExit(f"Add your own EXPERIMENT_LOG.md entry containing `{name}` and what it tests first.")
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." for c in name):
        raise SystemExit("Use a run name containing only letters, numbers, underscores, hyphens, or periods.")
    branch, commit, dirty = git_info()
    output_dir = f"/results/runs/{branch}/{name}"
    call = analyze.spawn(output_dir, samples_per_task, batch_size, seed,
                         dict(branch=branch, commit=commit, dirty=dirty))
    print(f"Spawned {call.object_id}; outputs: {output_dir}")
    call.get()
