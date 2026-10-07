"""Convert RLBench raw demonstrations to StarVLA's LeRobot v2 layout."""

from dataclasses import dataclass
import argparse
import json
from itertools import chain
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image
import yaml

from examples.simBenchmarks.RLbench.conversion.rlbench_reader import (
    RLBenchEpisode,
    RLBenchRawDataset,
)
from examples.simBenchmarks.RLbench.conversion.rlbench_schema import (
    ActionSpec,
    assert_action_dimension_contract,
)


@dataclass(frozen=True)
class ConversionResult:
    total_frames: int
    total_episodes: int
    action_dim: int
    state_dim: int
    output_root: Path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _modality_metadata(
    state_slices: Mapping[str, tuple[int, int]],
    camera_names: list[str],
    state_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    state = {
        name: {
            "start": start,
            "end": end,
            "original_key": "observation.state",
            "dtype": "float32",
        }
        for name, (start, end) in state_slices.items()
    }
    return {
        "state": state,
        "action": {
            "native": {
                "start": 0,
                "end": action_dim,
                "original_key": "action",
                "dtype": "float32",
            }
        },
        "video": {
            camera: {"original_key": f"observation.images.{camera}"}
            for camera in camera_names
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task"}
        },
        "state_dim": state_dim,
    }


def _info_metadata(
    episodes: int,
    frames: int,
    action_spec: ActionSpec,
    state_dim: int,
    camera_shapes: Mapping[str, tuple[int, ...]],
    fps: int,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": ["proprioception"],
        },
        "action": {
            "dtype": "float32",
            "shape": list(action_spec.action_shape),
            "names": ["native_action"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "task": {"dtype": "string", "shape": [1], "names": None},
        "observation.task_low_dim_state": {
            "dtype": "float32",
            "shape": [-1],
            "names": ["task_state"],
        },
    }
    for camera, shape in camera_shapes.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": list(shape),
            "names": ["height", "width", "rgb"],
        }
    return {
        "codebase_version": "v2.0",
        "robot_type": "rlbench_panda",
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
        "action_spec": action_spec.as_dict(),
    }


def _copy_image(array: np.ndarray, path: Path) -> dict[str, str]:
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected raw RGB uint8 HxWx3 image, received {array.shape} {array.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    return {"path": path.relative_to(path.parents[3]).as_posix()}


def _episode_frame_table(
    episode: RLBenchEpisode,
    output_root: Path,
    global_index: int,
    episode_id: int,
    fps: int,
) -> tuple[pd.DataFrame, dict[str, tuple[int, ...]]]:
    rows = []
    camera_shapes: dict[str, tuple[int, ...]] = {}
    for frame_index, frame in enumerate(episode.frames):
        row: dict[str, Any] = {
            "observation.state": frame.state,
            "action": frame.action,
            "timestamp": np.float32(frame_index / fps),
            "frame_index": np.int64(frame_index),
            "episode_index": np.int64(episode_id),
            "index": np.int64(global_index + frame_index),
            "task_index": np.int64(episode_id),
            "task": frame.task,
            "observation.task_low_dim_state": frame.task_low_dim_state,
        }
        for camera, image in frame.images.items():
            camera_shapes[camera] = tuple(int(size) for size in image.shape)
            image_path = (
                output_root
                / "images"
                / f"observation.images.{camera}"
                / f"episode_{episode_id:06d}"
                / f"{frame_index:06d}.png"
            )
            row[f"observation.images.{camera}"] = _copy_image(image, image_path)
        rows.append(row)
    return pd.DataFrame(rows), camera_shapes


def convert(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    action_mode: Mapping[str, Any],
    environment_action_shape: tuple[int, ...] | int,
    framework_action_dim: int,
    fps: int = 20,
) -> ConversionResult:
    """Convert raw RLBench episodes and fail before writing on dimension drift."""

    raw_root = Path(raw_root)
    output_root = Path(output_root)
    arm_type = str(action_mode["arm"]["type"])
    gripper_type = str(action_mode["gripper"]["type"])
    dataset = RLBenchRawDataset(
        raw_root,
        arm_type=arm_type,
        gripper_type=gripper_type,
    )
    episodes = iter(dataset)
    try:
        first_episode = next(episodes)
    except StopIteration:
        raise ValueError(f"No RLBench episodes found under {raw_root}")
    first_frame = first_episode.frames[0]
    action_shape = tuple(first_frame.action.shape)
    spec = ActionSpec(action_mode=action_mode, action_shape=action_shape)
    assert_action_dimension_contract(
        raw_action_dim=spec.action_dim,
        environment_action_shape=environment_action_shape,
        lerobot_action_shape=(len(first_frame.action),),
        framework_action_dim=framework_action_dim,
        spec=spec,
    )
    state_dim = len(first_frame.state)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = output_root / "data" / "chunk-000"
    data_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    camera_shapes: dict[str, tuple[int, ...]] = {}
    episodes_jsonl = []
    tasks: dict[str, int] = {}
    episode_count = 0
    for episode_id, episode in enumerate(chain((first_episode,), episodes)):
        if not episode.frames:
            raise ValueError(f"RLBench episode is empty: {episode.task_name}/{episode.episode_index}")
        if len(episode.frames[0].state) != state_dim:
            raise ValueError("State width changed between RLBench episodes")
        if any(frame.action.shape != action_shape for frame in episode.frames):
            raise ValueError("Action shape changed within an RLBench episode")
        task_index = tasks.setdefault(episode.frames[0].task, len(tasks))
        table, episode_cameras = _episode_frame_table(
            episode, output_root, total_frames, episode_id, fps
        )
        camera_shapes.update(episode_cameras)
        table.to_parquet(data_root / f"episode_{episode_id:06d}.parquet", index=False)
        length = len(table)
        episodes_jsonl.append(
            {
                "episode_index": episode_id,
                "tasks": [task_index],
                "length": length,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + length,
            }
        )
        total_frames += length
        episode_count += 1

    _write_json(output_root / "meta/action_spec.json", spec.as_dict())
    _write_json(
        output_root / "meta/info.json",
        {
            **_info_metadata(
                episode_count, total_frames, spec, state_dim, camera_shapes, fps
            ),
            "action_spec": spec.as_dict(),
        },
    )
    modality = _modality_metadata(
        first_frame.state_slices,
        sorted(camera_shapes),
        state_dim,
        spec.action_dim,
    )
    _write_json(output_root / "meta/modality.json", modality)
    (output_root / "meta").mkdir(exist_ok=True)
    with (output_root / "meta/episodes.jsonl").open("w", encoding="utf-8") as stream:
        for item in episodes_jsonl:
            stream.write(json.dumps(item) + "\n")
    with (output_root / "meta/tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task, index in sorted(tasks.items(), key=lambda pair: pair[1]):
            stream.write(json.dumps({"task_index": index, "task": task}) + "\n")
    return ConversionResult(
        total_frames=total_frames,
        total_episodes=episode_count,
        action_dim=spec.action_dim,
        state_dim=state_dim,
        output_root=output_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--environment-action-shape",
        type=int,
        required=True,
        help="RLBENCH Environment.action_shape() flattened dimension",
    )
    parser.add_argument("--framework-action-dim", type=int, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    result = convert(
        args.raw_root,
        args.output_root,
        action_mode=config["env"]["action_mode"],
        environment_action_shape=args.environment_action_shape,
        framework_action_dim=args.framework_action_dim,
        fps=args.fps,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "total_episodes": result.total_episodes,
                "total_frames": result.total_frames,
                "action_dim": result.action_dim,
                "state_dim": result.state_dim,
            }
        )
    )


if __name__ == "__main__":
    main()
