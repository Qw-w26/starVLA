"""TDD contracts for the RLBench to StarVLA LeRobot conversion."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from examples.simBenchmarks.RLbench.conversion.rlbench_reader import RLBenchRawDataset
from examples.simBenchmarks.RLbench.conversion.rlbench_schema import (
    ActionSpec,
    assert_action_dimension_contract,
)
from examples.simBenchmarks.RLbench.conversion.rlbench_to_lerobot import convert


def _write_fixture(root: Path) -> None:
    episode = root / "push_buttons" / "variation0" / "episodes" / "episode0"
    episode.mkdir(parents=True)
    observations = []
    for frame in range(2):
        observations.append(
            SimpleNamespace(
                joint_positions=np.arange(7, dtype=np.float64) + frame,
                joint_velocities=np.arange(7, dtype=np.float64) + 10 + frame,
                joint_forces=np.arange(7, dtype=np.float64) + 20 + frame,
                gripper_open=float(frame),
                gripper_joint_positions=np.array([0.1, 0.2], dtype=np.float64),
                gripper_touch_forces=np.arange(6, dtype=np.float64),
                gripper_pose=np.arange(7, dtype=np.float64),
                gripper_matrix=np.eye(4, dtype=np.float64),
                task_low_dim_state=np.arange(3, dtype=np.float64),
            )
        )
        for camera in ("front_rgb", "wrist_rgb"):
            camera_dir = episode / camera
            camera_dir.mkdir(exist_ok=True)
            Image.new("RGB", (3, 2), (frame, 2, 3)).save(camera_dir / f"{frame}.png")
    with (episode / "low_dim_obs.pkl").open("wb") as stream:
        pickle.dump(observations, stream)
    with (root / "push_buttons" / "variation0" / "variation_descriptions.pkl").open("wb") as stream:
        pickle.dump(["push the buttons"], stream)


class RLBenchConversionTest(unittest.TestCase):
    def test_action_dimension_contract_is_shared_by_all_layers(self):
        spec = ActionSpec(
            action_mode={
                "type": "MoveArmThenGripper",
                "arm": {"type": "JointVelocity"},
                "gripper": {"type": "Discrete"},
            },
            action_shape=(8,),
        )
        assert_action_dimension_contract(
            raw_action_dim=8,
            environment_action_shape=(8,),
            lerobot_action_shape=(4, 8),
            framework_action_dim=8,
            spec=spec,
        )

    def test_conversion_preserves_native_action_images_state_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            output_root = Path(temporary) / "lerobot"
            _write_fixture(raw_root)

            result = convert(
                raw_root,
                output_root,
                action_mode={
                    "type": "MoveArmThenGripper",
                    "arm": {"type": "JointVelocity"},
                    "gripper": {"type": "Discrete"},
                },
                environment_action_shape=(8,),
                framework_action_dim=8,
                fps=20,
            )

            self.assertEqual(result.total_frames, 2)
            self.assertEqual(result.action_dim, 8)
            self.assertEqual(result.state_dim, 53)
            action_spec = json.loads((output_root / "meta/action_spec.json").read_text())
            self.assertEqual(action_spec["source"], "RLBench")
            self.assertEqual(action_spec["action_shape"], [8])
            self.assertEqual(action_spec["action_dim"], 8)
            info = json.loads((output_root / "meta/info.json").read_text())
            self.assertEqual(info["action_spec"]["action_dim"], 8)
            modality = json.loads((output_root / "meta/modality.json").read_text())
            self.assertEqual(
                modality["video"]["front_rgb"]["original_key"],
                "observation.images.front_rgb",
            )
            self.assertEqual(modality["state"]["joint_positions"]["start"], 0)
            self.assertEqual(modality["state"]["joint_positions"]["end"], 7)

            parquet_files = list((output_root / "data").rglob("*.parquet"))
            self.assertEqual(len(parquet_files), 1)
            import pandas as pd

            frame = pd.read_parquet(parquet_files[0])
            self.assertEqual(np.asarray(frame["action"].iloc[0]).shape, (8,))
            self.assertEqual(np.asarray(frame["observation.state"].iloc[0]).shape, (53,))
            self.assertEqual(
                np.asarray(frame["observation.task_low_dim_state"].iloc[0]).shape,
                (3,),
            )
            self.assertEqual(frame["task"].tolist(), ["push the buttons", "push the buttons"])
            self.assertTrue(
                (output_root / "images" / "observation.images.front_rgb" / "episode_000000").exists()
            )

    def test_reader_derives_joint_velocity_action_without_reencoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            episode = next(iter(RLBenchRawDataset(root)))
            np.testing.assert_array_equal(
                episode.frames[0].action,
                np.array([10, 11, 12, 13, 14, 15, 16, 0], dtype=np.float32),
            )
            self.assertEqual(episode.frames[0].images["front_rgb"].dtype, np.uint8)
            self.assertEqual(episode.frames[0].state.shape, (53,))


if __name__ == "__main__":
    unittest.main()
