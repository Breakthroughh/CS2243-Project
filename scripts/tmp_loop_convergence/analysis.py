"""Checkpoint-only loop convergence. Dims: R loop boundaries (including input), B examples,
P action tokens, D hidden width, A action coordinates. No training or simulator rollouts.
"""
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F


@torch.inference_mode()
def capture_loops(policy, x_BPA, k_B, ctx_BLD):
    """Capture input to refinement (loop 0) and each loop's output, before the decoder."""
    states = []
    first = policy.blocks[policy.n_pre_layers]
    last = policy.blocks[policy.n_pre_layers + policy.n_loop_layers - 1]

    def capture_input(module, inputs):
        if not states:
            states.append(inputs[0].detach().float().cpu().clone())

    def capture_output(module, inputs, output):
        states.append(output.detach().float().cpu().clone())

    handles = [first.register_forward_pre_hook(capture_input), last.register_forward_hook(capture_output)]
    try:
        policy.denoise(x_BPA, k_B, ctx_BLD)
    finally:
        for handle in handles:
            handle.remove()
    if len(states) != policy.n_loops + 1:
        raise ValueError(f"Expected {policy.n_loops + 1} loop boundaries, got {len(states)}")
    return torch.stack(states)


def convergence(h_RBPD):
    """Tokenwise cosine averaged over P -> [R,B]; L2 distance relative to final [P,D] norm.
    Zero-norm comparisons are undefined (NaN), never silently treated as converged.
    """
    final_BPD = h_RBPD[-1]
    valid_RBP = (h_RBPD.norm(dim=-1) > 1e-8) & (final_BPD.norm(dim=-1) > 1e-8)
    cosine_RBP = F.cosine_similarity(h_RBPD, final_BPD[None], dim=-1).clamp(-1, 1)
    cosine_RB = cosine_RBP.masked_fill(~valid_RBP, float("nan")).nanmean(-1)
    norm_B = final_BPD.flatten(1).norm(dim=-1)
    distance_RB = (h_RBPD - final_BPD).flatten(2).norm(dim=-1) / norm_B.clamp_min(1e-8)
    distance_RB[:, norm_B <= 1e-8] = float("nan")
    return {"cosine": cosine_RB, "relative_l2": distance_RB}


def summarize(rows):
    """Pointwise mean +/- 1.96 SEM across examples; tokens are not independent samples."""
    groups = {}
    for row in rows:
        key = row["run"], row["n_loops"], row["diffusion_step"], row["loop"]
        groups.setdefault(key, []).append(row)
    summary = []
    for (run, loops, k, r), values in sorted(groups.items()):
        out = dict(run=run, n_loops=loops, diffusion_step=k, loop=r)
        for metric in ("cosine", "relative_l2"):
            x = torch.tensor([v[metric] for v in values], dtype=torch.float64)
            x = x[torch.isfinite(x)]
            out[f"{metric}_n"] = len(x)
            out[f"{metric}_mean"] = x.mean().item() if len(x) else float("nan")
            out[f"{metric}_sem"] = (x.std() / len(x) ** 0.5).item() if len(x) > 1 else float("nan")
        summary.append(out)
    return summary


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, timesteps, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {2: "#2563eb", 4: "#d97706", 6: "#059669"}
    runs = sorted({(s["n_loops"], s["run"]) for s in summary})
    for metric, ylabel, filename in (
        ("cosine", "Cosine similarity to final loop", "cosine_similarity"),
        ("relative_l2", "Relative L2 distance to final loop", "relative_l2"),
    ):
        fig, axes = plt.subplots(1, len(timesteps), figsize=(3.3 * len(timesteps), 3.8),
                                 sharey=True, squeeze=False)
        for ax, k in zip(axes[0], timesteps):
            for loops, run in runs:
                points = sorted((s for s in summary if s["run"] == run and s["diffusion_step"] == k),
                                key=lambda s: s["loop"])
                x = np.array([s["loop"] for s in points])
                y = np.array([s[f"{metric}_mean"] for s in points])
                error = 1.96 * np.array([s[f"{metric}_sem"] for s in points])
                ax.plot(x, y, "o-", color=colors[loops], label=f"{loops} loops", markersize=4)
                ax.fill_between(x, y - error, y + error, color=colors[loops], alpha=0.13)
            ax.set(title=f"Diffusion timestep {k}", xlabel="Refinement loop")
            ax.set_xticks(range(max(n for n, _ in runs) + 1))
            ax.grid(alpha=0.2)
        axes[0, 0].set_ylabel(ylabel)
        axes[0, -1].legend(frameon=False)
        fig.suptitle("Latest checkpoints: hidden action tokens before the decoder")
        fig.text(0.5, 0.01, "Loop 0 = input to shared blocks. Shading = +/- 1.96 SEM over examples; "
                 "each model uses its own final loop as reference.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.93))
        for extension in ("png", "pdf"):
            fig.savefig(Path(output_dir) / f"{filename}.{extension}", dpi=180)
        plt.close(fig)


@torch.inference_mode()
def analyze(run_specs, output_dir, timesteps=(0, 25, 50, 75, 99), samples_per_task=16, batch_size=16,
            seed=0, device="cuda"):
    """Use identical held-out windows and forward-diffused actions/noise for all checkpoints."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset
    from src.data import build_datasets

    if samples_per_task < 2 or batch_size < 1 or not timesteps or len(set(timesteps)) != len(timesteps):
        raise ValueError("Need >=2 samples per task, batch_size >=1, and distinct diffusion timesteps")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    configs = [OmegaConf.load(Path(spec["path"]) / "config.yaml") for spec in run_specs]
    data_cfg = OmegaConf.to_container(configs[0].data, resolve=True)
    if any(OmegaConf.to_container(cfg.data, resolve=True) != data_cfg for cfg in configs[1:]):
        raise ValueError("Checkpoint data configurations differ; paired comparison requires identical data")
    root = configs[0].data.root or configs[0].data.download_dir
    _, heldout = build_datasets(configs[0], root)
    generator = torch.Generator().manual_seed(seed)
    indices, sample_ids, offset = [], [], 0
    for task in heldout.datasets:
        if len(task) < samples_per_task:
            raise ValueError(f"{task.path} has fewer than {samples_per_task} held-out windows")
        chosen = torch.randperm(len(task), generator=generator)[:samples_per_task].sort().values.tolist()
        for i in chosen:
            demo, t, _ = task.index[i]
            indices.append(offset + i)
            sample_ids.append(dict(file=str(task.path), demo=demo, timestep=t))
        offset += len(task)
    noise_NPA = torch.randn(len(indices), configs[0].data.action_horizon, 7,
                            generator=torch.Generator().manual_seed(seed + 1))
    loader = DataLoader(Subset(heldout, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    manifest = dict(seed=seed, noise_seed=seed + 1, device=device, timesteps=list(timesteps), samples=sample_ids,
                    samples_per_task=samples_per_task, batch_size=batch_size,
                    reference="hidden state after final refinement loop, before post blocks/head",
                    input="held-out demo actions with fixed Gaussian forward-diffusion noise",
                    aggregation="cosine over D, mean over P, then mean and SEM across held-out windows",
                    uncertainty="descriptive +/-1.96 SEM; windows within a demo can be correlated",
                    checkpoints=[])
    rows = []
    for spec, cfg in zip(run_specs, configs):
        if cfg.policy.n_loops != spec["loops"]:
            raise ValueError(f"Unexpected loop count for {spec['path']}: {cfg.policy.n_loops}")
        if any(k < 0 or k >= cfg.policy.n_train_steps for k in timesteps):
            raise ValueError("Diffusion timestep outside checkpoint's noise schedule")
        torch.manual_seed(seed)
        # All ResNet weights/buffers are in the checkpoint; avoid an unnecessary ImageNet download.
        policy = instantiate(cfg.policy, pretrained_vision=False).to(device).eval()
        path = Path(spec["path"]) / "latest.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved_policy = checkpoint.get("policy_config")
        if saved_policy is not None and saved_policy != OmegaConf.to_container(cfg.policy, resolve=True):
            raise ValueError(f"Checkpoint policy configuration differs from config.yaml: {path}")
        policy.load_state_dict(checkpoint["policy"], strict=True)
        epoch, step = checkpoint["epoch"], checkpoint["step"]
        del checkpoint
        manifest["checkpoints"].append(dict(run=spec["run"], path=str(path), epoch=epoch, step=step,
                                            n_loops=policy.n_loops,
                                            config=OmegaConf.to_container(cfg, resolve=True)))
        cursor = 0
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            count = len(batch["action"])
            noise_BPA = noise_NPA[cursor:cursor + count].to(device)
            context_BLD = policy.encode(batch)
            for k in timesteps:
                a = policy.alpha_bar_K[k]
                noisy_BPA = a.sqrt() * batch["action"] + (1 - a).sqrt() * noise_BPA
                metrics = convergence(capture_loops(policy, noisy_BPA,
                                      torch.full((count,), k, device=device, dtype=torch.long), context_BLD))
                for r in range(policy.n_loops + 1):
                    for b in range(count):
                        rows.append(dict(run=spec["run"], n_loops=policy.n_loops, epoch=epoch, step=step,
                                         diffusion_step=k, loop=r, sample=cursor + b,
                                         **{key: value[r, b].item() for key, value in metrics.items()}))
            cursor += count
        print(f"Analyzed {spec['run']}: epoch={epoch}, step={step}, samples={cursor}")
        del policy
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    summary = summarize(rows)
    write_csv(output_dir / "per_sample.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    plot(summary, timesteps, output_dir)
    print(f"Saved plots and source measurements to {output_dir}")
