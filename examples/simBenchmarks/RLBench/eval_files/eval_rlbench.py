# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""RLBench closed-loop evaluator.

Run AFTER starting the starVLA policy server (see eval_rlbench.sh / README).
Connects an :class:`RLBenchModelClient`, rolls out episodes in CoppiuliaSim>
RLBench, and reports per-task success rate.

The policy is conditioned on the front camera ("primary_image") + wrist camera,
and predicts absolute 8-DoF joint + gripper targets that are executed with
RLBench's ``JointPositionActionMode``.

Usage::

    bash examples/simBenchmarks/RLBench/eval_files/eval_rlbench.sh \
        --tasks reach_target,open_drawer --trials 10
    # or directly:
    python examples/simBenchmarks/RLBench/eval_files/eval_rlbench.py \
        --host 127.0.0.1 --port 6694 \
        --tasks reach_target,open_drawer --trials 10
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6694)
    p.add_argument("--unnorm-key", default="rlbench_panda")
    p.add_argument("--tasks", default="reach_target",
                   help="comma-separated RLBench task class names")
    p.add_argument("--trials", type=int, default=10, help="trials per task")
    p.add_argument("--max-steps", type=int, default=200,
                   help="max env steps per episode")
    p.add_argument("--image-size", nargs=2, type=int, default=None,
                   help="resize both cameras to H W before sending")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = _parse_args()
    np.random.seed(args.seed)
    os.environ.setdefault("DISPLAY", ":0")

    from rlbench import Environment, tasks
    from rlbench.action_modes.action_mode import JointPositionActionMode
    from rlbench import ObservationConfig
    from model2rlbench_interface import RLBenchModelClient

    obs_cfg = ObservationConfig()
    obs_cfg.set_all(False)
    for cam in (obs_cfg.front_camera, obs_cfg.wrist_camera):
        cam.set_all(True)
        cam.rgb = True; cam.depth = False; cam.point_cloud = False; cam.mask = False
    obs_cfg.joint_positions = True
    obs_cfg.gripper_open = True

    env = Environment(action_mode=JointPositionActionMode(),
                      obs_config=obs_cfg, headless=False)
    env.launch()
    print("[eval] CoppeliaSim launched", flush=True)

    client = RLBenchModelClient(host=args.host, port=args.port,
                                unnorm_key=args.unnorm_key,
                                image_size_hw=tuple(args.image_size) if args.image_size else None)
    # soft-check that the server was trained for this robot contract
    avail_unnorm = client.client.get_server_metadata().get("available_unnorm_keys")
    if avail_unnorm and args.unnorm_key not in avail_unnorm:
        print(f"[eval][WARNING] unnorm_key {args.unnorm_key!r} not in server "
              f"available_unnorm_keys={avail_unnorm}", flush=True)

    def _snake_to_camel(name: str) -> str:
        return "".join(part.capitalize() for part in name.split("_") if part)

    names = [_snake_to_camel(t.strip()) for t in args.tasks.split(",") if t.strip()]
    results = {}
    for name in names:
        task_cls = getattr(tasks, name)
        task = env.get_task(task_cls)
        desc = task.get_language_form()[0]
        successes = 0
        for trial in range(args.trials):
            client.reset()
            instr, obs = task.reset()
            if isinstance(instr, (list, tuple)):
                instr = instr[0]
            for step in range(args.max_steps):
                model_obs = {
                    "images": {"primary_image": obs.front_rgb, "wrist_image": obs.wrist_rgb},
                    "lang": instr,
                }
                action = client.step(model_obs, step)           # (8,)
                obs, reward, terminated = task.step(action.astype(np.float32))
                if reward > 0 or terminated:
                    if reward > 0:
                        successes += 1
                    break
            print(f"[eval] {name} trial {trial}: {'SUCCESS' if reward > 0 else 'fail'}", flush=True)
        results[name] = successes / args.trials
        print(f"[eval] {name}: {successes}/{args.trials} = {results[name]:.2f}", flush=True)

    env.shutdown()
    print("\n=== RLBench results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()