# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""RLBench env-side adapter (thin client for the starVLA websocket server).

The starVLA policy *server* (`deployment/model_server/server_policy.py`) already
unnormalizes actions and ships model-invariant fields (`action_chunk_size`,
`available_unnorm_keys`) at handshake, so this client does not touch
`dataset_statistics.json` and performs no normalization.

It mirrors ``examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py``
but returns the *raw* 8-DoF chunk (7 joint positions + gripper) that
``eval_rlbench.py`` feeds to CoppiuliaSim's ``JointPosition`` action mode.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

# On-disk RLBench camera contract -> model keys. Must match training order
# (see train_files/data_registry/data_config.py video_keys).
MODEL_IMAGE_KEYS = ("primary_image", "wrist_image")
# RLBench Observation attribute per model key (front / wrist RGB).
RLBENCH_IMAGE_ATTR = {"primary_image": "front_rgb", "wrist_image": "wrist_rgb"}


class RLBenchModelClient:
    """Client that returns an unnormalized (action_chunk_size, 8) joint chunk."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 6694,
        unnorm_key: str = "rlbench_panda",
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        image_size_hw: Optional[Sequence[int]] = None,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self.image_size_hw = tuple(image_size_hw) if image_size_hw else None
        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        print(f"[rlbench] server metadata: {meta}", flush=True)
        print(f"[rlbench] action_chunk_size={self.action_chunk_size} "
              f"unnorm_key={unnorm_key}", flush=True)

        self.raw_actions: Optional[np.ndarray] = None  # cached (T, 8) chunk

    def reset(self) -> None:
        """Called once per evaluation episode."""
        self.raw_actions = None

    def _rebuild_example(self, obs: dict) -> dict:
        images = []
        for key in MODEL_IMAGE_KEYS:
            img = np.asarray(obs["images"][key])
            if self.image_size_hw and img.shape[:2] != self.image_size_hw:
                img = np.asarray(Image.fromarray(img).resize(
                    (self.image_size_hw[1], self.image_size_hw[0]), Image.BILINEAR))
            images.append(img)
        return {"image": images, "lang": obs["lang"]}

    def step(self, obs: dict, step: int) -> np.ndarray:
        """Given env obs + step, return ONE action (8,) to execute now.

        The cached chunk is refreshed every ``action_chunk_size`` steps.
        """
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            vla_input = {
                "examples": [self._rebuild_example(obs)],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            try:
                actions_batch = response["data"]["actions"]  # (B, T, D) unnormalized
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: "
                    f"keys={list(response.get('data', {}).keys())}, response={response}")
            self.raw_actions = np.asarray(actions_batch)[0]  # (T, 8)

        return self.raw_actions[step % self.action_chunk_size]