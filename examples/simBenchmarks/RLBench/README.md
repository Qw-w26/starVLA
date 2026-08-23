# RLBench

[RLBench](https://github.com/stepjam/RLBench) is a CoppiuliaSim-based
tabletop manipulation benchmark (7-DoF Franka/Panda + gripper, 100+ tasks).
StarVLA trains on RLBench demonstrations and evaluates a policy back inside
CoppiuliaSim. Because RLBench ships no ready-to-train dataset, this directory
provides **generation** (CoppiuliaSim → LeRobot v2.1), **training**
(auto-discovered registry + recipe), and **evaluation** (closed-loop success
rate in CoppiuliaSim) for it.

| Item | Value |
|---|---|
| Robot | Franka/Panda, 7-DoF + 1 gripper (8-DoF action) |
| Cameras | Front (`primary_image`) + wrist (`wrist_image`), 256×256 |
| State / Action | Absolute joint position (8-DoF), `abs_qpos`, min/max norm |
| Action horizon | 8 |
| Dataset format | LeRobot v2.1 *video* (`rlbench_lerobot/`) |

## Directory layout

```
RLBench/
  README.md
  tools/collect_data.py           # RLBench demos -> LeRobot v2.1 dataset
  train_files/
    data_registry/data_config.py  # auto-discovered starVLA registry
    modality.json                 # gr00t schema (copy into meta/modality.json)
    starvla_cotrain_rlbench.yaml  # training config
    run_rlbench_train.sh          # training launcher
  eval_files/
    run_policy_server.sh          # start the starVLA policy server
    eval_rlbench.sh               # eval launcher
    eval_rlbench.py               # closed-loop evaluator
    model2rlbench_interface.py    # env-side policy client
    install_rlbench.sh            # env install (headless-safe)
```

## ⚠️ Environment: CoppeliaSim MUST be **4.1.0** (not 4.10.0)

RLBench 1.2.0 / PyRep 4.1.0.3 are built against **CoppeliaSim 4.1.0**. A 4.10.0
build launches and even loads the scene, but RLBench's `Scene` init crashes with
native heap corruption (`free(): invalid next size`) due to the ABI drift.

```bash
# one-time
bash examples/simBenchmarks/RLBench/eval_files/install_rlbench.sh

# each shell that launches RLBench — exactly ONE CoppeliaSim on LD_LIBRARY_PATH
export COPPELIASIM_ROOT=$HOME/CoppeliaSim_4_1
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT/platforms
export DISPLAY=:0        # or a GLX-capable virtual X
```

Headless note: the camera renderer needs a real GL context. On a box with no
usable GL (GPU-less WSL, no `/dev/dri`) it segfaults (`signal 11`,
`QOpenGLContext`). Run data generation / eval on a GPU or headed machine (e.g. a
GPU cluster node), or bring up `xvfb` + Mesa GLX.

## 1. Collect training data

```bash
python examples/simBenchmarks/RLBench/tools/collect_data.py \
  --data-root playground/Datasets/RLBENCH \
  --tasks reach_target,open_drawer,close_box,stack_blocks \
  --episodes 10 \
  --headless
```

Writes `playground/Datasets/RLBENCH/rlbench_lerobot/` with `meta/`,
`data/chunk-000/episode_*.parquet`, and `videos/chunk-000/...` (see
`tools/collect_data.py` header for exact layout). `spotcheck/` gets one frame
per episode for a quick visual sanity check. The dataset is registered by the
auto-discovery in `train_files/data_registry/data_config.py` as the
`rlbench_all` mixture.

Train a real policy with far more episodes than the smoke 10; typical RLBench
VLA runs use 50–100 demonstration episodes per task, across many tasks.

## 2. Train

The recipe trains the VLM + interface + action model end to end (default is
QwenPI_v3; change `Framework_name` in the launcher).

```bash
bash examples/simBenchmarks/RLBench/train_files/run_rlbench_train.sh
```

The registry is auto-discovered from
`train_files/data_registry/data_config.py`; points `datasets.vla_data.data_mix`
to `rlbench_all`, `data_root_dir` to the parent of `rlbench_lerobot`.

## 3. Evaluate (two terminals)

Terminal 1 — start the starVLA policy server for the RLBench checkpoint:

```bash
CKPT=/path/to/rlbench/checkpoints/steps_NN_pytorch_model.pt \
GPU_ID=0 PORT=6694 \
bash examples/simBenchmarks/RLBench/eval_files/run_policy_server.sh
```

Terminal 2 — run the closed-loop evaluator (CoppiuliaSim side):

```bash
export COPPELIASIM_ROOT=...        # as above
TASKS=reach_target,open_drawer TRIALS=10 \
bash examples/simBenchmarks/RLBench/eval_files/eval_rlbench.sh
```

Success rate per task is printed (and breaks on episode **terminated**).

## References

- RLBench: <https://github.com/stepjam/RLBench> (requires CoppeliaSim 4.1.0)
- PyRep: <https://github.com/stepjam/PyRep>
- starVLA benchmark style cribbed from
  [`examples/simBenchmarks/RoboDojo`](../RoboDojo/README.md) and
  [`examples/simBenchmarks/LIBERO`](../LIBERO/README.md).