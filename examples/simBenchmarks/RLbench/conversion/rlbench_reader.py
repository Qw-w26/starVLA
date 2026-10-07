"""Reader for the directory layout emitted by RLBench dataset_generator."""

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterator, Mapping

import numpy as np
from PIL import Image

from .rlbench_schema import action_from_observation, state_from_observation


@dataclass(frozen=True)
class RLBenchFrame:
    action: np.ndarray
    state: np.ndarray
    state_slices: Mapping[str, tuple[int, int]]
    images: Mapping[str, np.ndarray]
    task_low_dim_state: np.ndarray | None
    task: str


@dataclass(frozen=True)
class RLBenchEpisode:
    task_name: str
    variation: int
    episode_index: int
    frames: tuple[RLBenchFrame, ...]


def _numeric_pngs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"), key=lambda path: int(path.stem))


class RLBenchRawDataset:
    """Iterate raw demonstrations without importing RLBench or CoppeliaSim."""

    def __init__(
        self,
        root: str | Path,
        *,
        arm_type: str = "JointVelocity",
        gripper_type: str = "Discrete",
    ) -> None:
        self.root = Path(root)
        self.arm_type = arm_type
        self.gripper_type = gripper_type
        if not self.root.is_dir():
            raise FileNotFoundError(f"RLBench raw dataset does not exist: {self.root}")

    def __iter__(self) -> Iterator[RLBenchEpisode]:
        for task_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            for variation_dir in sorted(task_dir.glob("variation*")):
                descriptions = self._descriptions(variation_dir)
                for episode_dir in sorted((variation_dir / "episodes").glob("episode*")):
                    yield self._read_episode(
                        task_dir.name,
                        int(variation_dir.name.removeprefix("variation")),
                        int(episode_dir.name.removeprefix("episode")),
                        episode_dir,
                        descriptions,
                    )

    @staticmethod
    def _descriptions(variation_dir: Path) -> list[str]:
        path = variation_dir / "variation_descriptions.pkl"
        if not path.exists():
            return [variation_dir.parent.name.replace("_", " ")]
        with path.open("rb") as stream:
            descriptions = pickle.load(stream)
        return [str(item) for item in descriptions] or [variation_dir.parent.name]

    def _read_episode(
        self,
        task_name: str,
        variation: int,
        episode_index: int,
        episode_dir: Path,
        descriptions: list[str],
    ) -> RLBenchEpisode:
        with (episode_dir / "low_dim_obs.pkl").open("rb") as stream:
            observations = pickle.load(stream)
        camera_paths = {
            camera_dir.name: _numeric_pngs(camera_dir)
            for camera_dir in sorted(episode_dir.iterdir())
            if camera_dir.is_dir() and _numeric_pngs(camera_dir)
            and camera_dir.name.endswith("_rgb")
        }
        frames = []
        task = descriptions[0]
        for index, observation in enumerate(observations):
            state, state_slices = state_from_observation(observation)
            images = {}
            for camera, paths in camera_paths.items():
                if index >= len(paths):
                    raise ValueError(
                        f"Camera {camera} has {len(paths)} frames but episode has "
                        f"{len(observations)} observations: {episode_dir}"
                    )
                with Image.open(paths[index]) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                images[camera] = np.ascontiguousarray(rgb)
            frames.append(
                RLBenchFrame(
                    action=action_from_observation(observation, self.arm_type, self.gripper_type),
                    state=state,
                    state_slices=state_slices,
                    images=images,
                    task_low_dim_state=(
                        None
                        if getattr(observation, "task_low_dim_state", None) is None
                        else np.asarray(observation.task_low_dim_state, dtype=np.float32).reshape(-1)
                    ),
                    task=task,
                )
            )
        return RLBenchEpisode(task_name, variation, episode_index, tuple(frames))
