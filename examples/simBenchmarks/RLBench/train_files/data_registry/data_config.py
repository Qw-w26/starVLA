"""RLBench — data config, embodiment tags, and mixtures.

Auto-discovered by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``.
This file must export ``ROBOT_TYPE_CONFIG_MAP`` and ``DATASET_NAMED_MIXTURES``.

The dataset is produced by ``tools/collect_data.py`` (RLBench raw demos ->
LeRobot v2.1 video format). Camera / state / action key names below MUST match
what the converter wrote into ``meta/modality.json`` and the per-episode
parquet columns.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RLBenchPandaDataConfig:
    """7-DoF Panda joint-position + 1-DoF gripper, 2 cameras (front + wrist).

    State  = current joint positions (7) + gripper (1)                        -> 8D
    Action = same absolute joint target                                       -> 8D
    Frame  = front (primary) + wrist RGB, 256x256, LeRobot video modality
    Norm   = per-key min/max -> [-1, 1] (matches the eval interface)
    """

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.primary_image",   # front camera -> original_key observation.images.image
        "video.wrist_image",     # wrist camera  -> original_key observation.images.wrist_image
    ]
    state_keys = [
        "state.joint_positions",  # 7 arm joints
        "state.gripper",          # 1 gripper
    ]
    action_keys = [
        "action.joint_positions",  # 7 arm joints (absolute)
        "action.gripper",          # 1 gripper
    ]
    state_key_dims = {
        "state.joint_positions": 7,
        "state.gripper": 1,
    }
    action_key_dims = {
        "action.joint_positions": 7,
        "action.gripper": 1,
    }
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]              # current frame only
    action_indices = list(range(ACTION_HORIZON := 8))  # predict t .. t+7

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
        state_modes = {k: "min_max" for k in self.state_keys}
        action_modes = {k: "min_max" for k in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(apply_to=self.state_keys, normalization_modes=state_modes),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(apply_to=self.action_keys, normalization_modes=action_modes),
            ]
        )


# robot_type string shared by: DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP,
# the eval interface (model2rlbench_interface.py) and dataset_statistics.json.
_RLBENCH_ROBOT = "rlbench_panda"

ROBOT_TYPE_CONFIG_MAP = {
    _RLBENCH_ROBOT: RLBenchPandaDataConfig(),
}

DATASET_NAMED_MIXTURES = {
    # mixture_name -> (dataset_subdir /*relative to vla_data.data_root_dir*/,
    #                  weight, robot_type_string)
    "rlbench_all": [
        ("rlbench_lerobot", 1.0, _RLBENCH_ROBOT),
    ],
}