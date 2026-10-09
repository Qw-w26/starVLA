"""StarVLA data configuration for native-action RLBench datasets."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RLBenchJointVelocityDataConfig:
    """Fixed schema for JointVelocity + Discrete Panda demonstrations."""

    def __init__(self, action_horizon: int):
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        self.action_indices = list(range(action_horizon))

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.front_rgb",
        "video.left_shoulder_rgb",
        "video.right_shoulder_rgb",
        "video.wrist_rgb",
        "video.overhead_rgb",
    ]
    state_keys = [
        "state.joint_positions",
        "state.joint_velocities",
        "state.joint_forces",
        "state.gripper_open",
        "state.gripper_joint_positions",
        "state.gripper_touch_forces",
        "state.gripper_pose",
        "state.gripper_matrix",
    ]
    action_keys = ["action.native"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        key: "min_max" for key in self.state_keys
                    },
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.native": "min_max"},
                ),
            ]
        )

