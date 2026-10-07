#!/usr/bin/env bash
set -euo pipefail

# Resolve the native RLBench action dimension from a launched scene, then pass
# it to an otherwise ordinary StarVLA training invocation. Dataset actions and
# normalization statistics must have been generated with this same config.
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
RLBENCH_ENV_CONFIG="${RLBENCH_ENV_CONFIG:-$STARVLA_DIR/examples/simBenchmarks/RLbench/configs/joint_velocity.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:?Set TRAIN_CONFIG to a StarVLA training YAML.}"

cd "$STARVLA_DIR"
ACTION_DIM="$({
  python examples/simBenchmarks/RLbench/eval_files/resolve_action_dim.py --config "$RLBENCH_ENV_CONFIG"
} | sed -n 's/^RLBENCH_ACTION_DIM=//p' | tail -n 1)"
if [[ ! "$ACTION_DIM" =~ ^[1-9][0-9]*$ ]]; then
  echo "Failed to resolve a numeric RLBench action dimension from ${RLBENCH_ENV_CONFIG}" >&2
  exit 1
fi
echo "Resolved RLBench action_dim=${ACTION_DIM} from ${RLBENCH_ENV_CONFIG}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml "$TRAIN_CONFIG" \
  --framework.action_model.action_dim "$ACTION_DIM" \
  "$@"
