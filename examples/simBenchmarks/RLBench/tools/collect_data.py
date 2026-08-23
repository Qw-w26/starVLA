#!/usr/bin/env python
"""Collect RLBench demonstrations and write them as a starVLA-consumable
LeRobot v2.1 *video* dataset.

RLBench itself stores no ready-to-train LeRobot export, so this tool drives
CoppeliaSim + RLBench live and writes episodes in exactly the layout the
starVLA loader expects (mirrors LIBERO's ``*_lerobot`` layout):

    <data_root>/rlbench_lerobot/
        meta/info.json
        meta/modality.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        data/chunk-000/episode_000000.parquet   (one row per step: state/action/task)
        videos/chunk-000/observation.images.image/episode_000000.mp4        (front)
        videos/chunk-000/observation.images.wrist_image/episode_000000.mp4  (wrist)

Modality/action contract (see train_files/data_registry/data_config.py):

    state  = [7 joint positions, gripper]           -> 8D observation.state
    action = [7 joint positions, gripper] (target)  -> 8D action
    video.primary_image <- front  camera
    video.wrist_image   <- wrist  camera

Environment prerequisites (the RLBench docs' CoppiuliaSim 4.1.0 + PyRep + rlbench):

    export COPPELIASIM_ROOT=~/CoppeliaSim_4_1          # exactly ONE version, must be 4.1.0
    export LD_LIBRARY_PATH=$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT/platforms
    export DISPLAY=:0                                   # or a GLX-capable virtual X

Usage::

    python tools/collect_data.py \
        --data-root /path/to/playground/Datasets/RLBENCH \
        --tasks reach_target,open_drawer,close_box \
        --episodes-per-task 10 \
        --headless

Options:
    --tasks             comma list of RLBench task names (default: reach_target)
    --episodes          episodes per task (default: 10)
    --image-size        encoding resolution (square), default 256
    --fps               nominal video fps (default 20)
    --seed              numpy/sim seed
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

IMAGES_KEYS = ("observation.images.image", "observation.images.wrist_image")
VIEW_KEY_FROM_RLBENCH = {"primary": "front_rgb", "wrist": "wrist_rgb"}
DATASET_DIR_NAME = "rlbench_lerobot"


def snake_to_camel(name: str) -> str:
    """'reach_target' -> 'ReachTarget' (RLBench task classes are PascalCase)."""
    return "".join(part.capitalize() for part in name.split("_") if part)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./playground/Datasets/RLBENCH",
                   help="parent dir; dataset is written under <data-root>/rlbench_lerobot/")
    p.add_argument("--tasks", default="reach_target",
                   help="comma-separated RLBench task class names")
    p.add_argument("--episodes", type=int, default=10, help="episodes per task")
    p.add_argument("--image-size", type=int, default=256,
                   help="encode both cameras at this square resolution")
    p.add_argument("--fps", type=int, default=20, help="nominal video fps")
    p.add_argument("--headless", action="store_true", default=False,
                   help="run CoppeliaSim headless (default False; GUI mode is the "
                        "most reliable way to get a working OpenGL context on "
                        "Xvfb/GPU-less hosts)")
    p.add_argument("--gui", action="store_true", default=False,
                   help="explicitly run with the CoppeliaSim GUI (alias for not headless)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_observation_config():
    """Front + wrist RGB only, plus the joint/gripper state we need."""
    from rlbench import ObservationConfig

    cfg = ObservationConfig()
    cfg.set_all(False)
    for cam in (cfg.front_camera, cfg.wrist_camera):
        cam.set_all(True)
        cam.width, cam.height = 640, 480
        cam.rgb = True
        cam.depth = False
        cam.point_cloud = False
        cam.mask = False
    cfg.joint_positions = True
    cfg.gripper_open = True
    cfg.gripper_pose = True
    cfg.gripper_joint_positions = True
    cfg.joint_forces = False
    cfg.joint_velocities = False
    return cfg


def encode_video(frames, path: Path, fps: int):
    """Write RGB frames (H,W,3 uint8) to an H.264 mp4 via PyAV."""
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames[0].shape[:2]
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = W, H
    stream.pix_fmt = "yuv420p"
    for img in frames:
        img_yuv = av.video.VideoFrame.from_ndarray(
            np.ascontiguousarray(img), format="rgb24")
        for packet in stream.encode(img_yuv):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def write_episode_parquet(ep_dir: Path, ep_index: int, task_index: int,
                          states, actions, fps: int):
    import pandas as pd
    n = len(states)
    ts = np.arange(n, dtype=np.float32) / fps
    df = pd.DataFrame({
        "observation.state": list(states),
        "action": list(actions),
        "timestamp": list(ts),
        "frame_index": list(np.arange(n, dtype=np.int64)),
        "episode_index": np.full(n, ep_index, dtype=np.int64),
        "index": np.arange(n, dtype=np.int64),
        "task_index": np.full(n, task_index, dtype=np.int64),
    })
    ep_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ep_dir / f"episode_{ep_index:06d}.parquet")


def main():
    args = _parse_args()
    np.random.seed(args.seed)

    os.environ.setdefault("DISPLAY", ":0")
    from rlbench import Environment, tasks
    from rlbench.action_modes.action_mode import JointPositionActionMode

    obs_cfg = build_observation_config()
    headless = bool(args.headless and not args.gui)
    env = Environment(action_mode=JointPositionActionMode(),
                      obs_config=obs_cfg,
                      headless=headless)
    print("[collect] launching CoppeliaSim (RLBench)...", flush=True)
    env.launch()
    print("[collect] launched", flush=True)

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # RLBench task classes are PascalCase (e.g. ReachTarget); the CLI accepts
    # snake_case (reach_target). Warm-validate the class names.
    for i, name in enumerate(task_names):
        klass = snake_to_camel(name)
        if not hasattr(tasks, klass):
            raise SystemExit(f"[collect] unknown RLBench task: {name!r} (->{klass}). "
                             f"Available under rlbench.tasks, e.g. ReachTarget")
        task_names[i] = klass

    data_root = Path(args.data_root)
    ds = data_root / DATASET_DIR_NAME
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    (ds / "data").mkdir(parents=True, exist_ok=True)
    (ds / "videos").mkdir(parents=True, exist_ok=True)

    episodes_rows = []
    task_index_map = {}
    ep_index = 0
    chunk = "chunk-000"

    for name in task_names:
        task_cls = getattr(tasks, name)
        task = env.get_task(task_cls)
        # language instruction (RLBench 1.2.0 exposes it from task.reset())
        _desc, _obs = task.reset()
        language = _desc[0] if isinstance(_desc, (list, tuple)) else _desc
        if language not in task_index_map:
            task_index_map[language] = len(task_index_map)

        print(f"[collect] task {name!r} / '{language}'  ({args.episodes} eps)", flush=True)
        demos = task.get_demos(amount=args.episodes, live_demos=True)

        for demo in demos:
            front = [o.front_rgb for o in demo]
            wrist = [o.wrist_rgb for o in demo]
            joints = np.array([o.joint_positions for o in demo], dtype=np.float32)
            greys = np.array([float(o.gripper_open) for o in demo], dtype=np.float32)
            states = np.concatenate([joints, greys[:, None]], axis=1)          # N,8
            actions = np.concatenate(
                [np.roll(joints, -1, axis=0), np.roll(greys, -1)[:, None]], axis=1)
            actions[-1] = states[-1]                                            # last target = last state

            # trim to a common frame length
            n = min(len(front), len(wrist), len(joints))
            front, wrist = front[:n], wrist[:n]
            states, actions = states[:n], actions[:n]

            # encode videos (front -> primary, wrist -> wrist)
            def _to_rgb(fr):
                img = np.asarray(fr)
                if img.shape[0] != args.image_size or img.shape[1] != args.image_size:
                    img = _resize(img, args.image_size, args.image_size)
                return img.astype(np.uint8)

            ep_dir = ds / "data" / chunk
            write_episode_parquet(ep_dir, ep_index, task_index_map[language], states, actions, args.fps)
            encode_video([_to_rgb(f) for f in front],
                         ds / "videos" / chunk / IMAGES_KEYS[0] / f"episode_{ep_index:06d}.mp4",
                         args.fps)
            encode_video([_to_rgb(f) for f in wrist],
                         ds / "videos" / chunk / IMAGES_KEYS[1] / f"episode_{ep_index:06d}.mp4",
                         args.fps)

            episodes_rows.append({
                "episode_index": ep_index,
                "tasks": [language],
                "length": n,
            })

            # dump one frame for headless manual spot-check
            sp = ds / "spotcheck"
            sp.mkdir(exist_ok=True)
            Image.fromarray(_to_rgb(front[n // 2])).save(sp / f"ep{ep_index:04d}_primary.jpg")
            ep_index += 1
            print(f"[collect]   episode {ep_index-1} frames={n}", flush=True)

    env.shutdown()

    # ---- metadata ----
    tasks_jsonl = ds / "meta" / "tasks.jsonl"
    with open(tasks_jsonl, "w") as f:
        for i, lang in enumerate(task_index_map):
            f.write(json.dumps({"task_index": i, "task": lang}) + "\n")

    with open(ds / "meta" / "episodes.jsonl", "w") as f:
        for row in episodes_rows:
            f.write(json.dumps(row) + "\n")

    # copy the checked-in modality schema
    here = Path(__file__).resolve().parent
    modality_src = here / ".." / "train_files" / "modality.json"
    (ds / "meta" / "modality.json").write_text(modality_src.read_text())

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(episodes_rows),
        "total_frames": sum(r["length"] for r in episodes_rows),
        "total_tasks": len(task_index_map),
        "total_videos": 2 * len(episodes_rows),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image": _video_feature(args.image_size, args.fps),
            "observation.images.wrist_image": _video_feature(args.image_size, args.fps),
            "observation.state": _lowdim_feature(8, ["joint" + str(i) for i in range(7)] + ["gripper"]),
            "action": _lowdim_feature(8, ["joint" + str(i) for i in range(7)] + ["gripper"]),
            "timestamp": _scalar_feature("float32"),
            "frame_index": _scalar_feature("int64"),
            "episode_index": _scalar_feature("int64"),
            "index": _scalar_feature("int64"),
            "task_index": _scalar_feature("int64"),
        },
    }
    (ds / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"[collect] DONE. dataset at {ds} ({len(episodes_rows)} episodes, "
          f"{len(task_index_map)} tasks). Spot-check frames in {ds/'spotcheck'}",
          flush=True)


def _video_feature(size, fps):
    return {
        "dtype": "video",
        "shape": [size, size, 3],
        "names": ["height", "width", "rgb"],
        "info": {"video.height": size, "video.width": size,
                 "video.codec": "h264", "video.pix_fmt": "yuv420p",
                 "video.is_depth_map": False, "video.fps": fps,
                 "video.channels": 3, "has_audio": False},
    }


def _lowdim_feature(dim, names):
    return {"dtype": "float32", "shape": [dim], "names": {"motors": names}}


def _scalar_feature(dtype):
    return {"dtype": dtype, "shape": [1], "names": None}


def _resize(img, w, h):
    return np.asarray(Image.fromarray(np.asarray(img)).resize((w, h), Image.Resampling.LANCZOS))


if __name__ == "__main__":
    main()