"""A simulator smoke test, not a learned policy or benchmark evaluation."""
import json
import os
from pathlib import Path


def configure_libero(root="/opt/LIBERO"):
    # LIBERO otherwise prompts for configuration on first import.
    import yaml
    base = Path(root) / "libero/libero"
    folder = Path(os.environ.get("LIBERO_CONFIG_PATH", "/root/.libero"))
    folder.mkdir(parents=True, exist_ok=True)
    datasets = Path("/tmp/libero-datasets")
    datasets.mkdir(exist_ok=True)
    config = dict(benchmark_root=str(base), bddl_files=str(base / "bddl_files"),
                  init_states=str(base / "init_files"), assets=str(base / "assets"),
                  datasets=str(datasets))
    (folder / "config.yaml").write_text(yaml.safe_dump(config))


def list_tasks(suite_name):
    configure_libero()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    return [{"task_id": i, "name": suite.get_task(i).name,
             "instruction": suite.get_task(i).language}
            for i in range(suite.get_num_tasks())]


def rollout(output, suite_name="libero_spatial", task_id=0, init_state_id=0,
            steps=60, seed=0):
    configure_libero()
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    if not 0 <= task_id < suite.get_num_tasks():
        raise ValueError("task_id is outside this suite")
    states = suite.get_task_init_states(task_id)
    if not 0 <= init_state_id < len(states):
        raise ValueError("init_state_id is outside the available initial states")
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                             camera_heights=256, camera_widths=256)
    summary = dict(suite=suite_name, task_id=task_id, task_name=task.name,
                   instruction=task.language, init_state_id=init_state_id,
                   seed=seed, controller="OSC_POSE", policy="no_op",
                   note="Installation smoke test; not a policy benchmark.")
    try:
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(states[init_state_id])
        # Six zero pose commands plus an open-gripper command.
        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        success = False
        with imageio.get_writer(output / "rollout.mp4", fps=20) as video:
            for step in range(steps):
                obs, reward, done, info = env.step(action)
                # Display-only flip; policy preprocessing must be model-specific.
                frame = np.ascontiguousarray(obs["agentview_image"][::-1])
                video.append_data(frame)
                success = success or bool(env.check_success())
                if done:
                    break
        imageio.imwrite(output / "agentview.png", frame)
        imageio.imwrite(output / "wrist.png", obs["robot0_eye_in_hand_image"][::-1])
        np.savez_compressed(output / "last_observation.npz", **{
            key: value for key, value in obs.items() if isinstance(value, np.ndarray)})
        summary.update(steps_executed=step + 1, success=success,
                       observation_shapes={k: list(np.shape(v)) for k, v in obs.items()})
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()
