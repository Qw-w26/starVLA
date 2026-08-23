#!/usr/bin/env bash
# RLBench evaluator launcher.
# Requires the policy server already running (eval_files/run_policy_server.sh).
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
RLBENCH_PYTHON="${RLBENCH_PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASKS="${TASKS:-reach_target,open_drawer,close_box}"
TRIALS="${TRIALS:-10}"
MAX_STEPS="${MAX_STEPS:-200}"

# The RLBench (CoppiuliaSim) side — see README / collect_data.py for the env vars.
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:?set to COPPELIASIM_ROOT=/path/to/CoppeliaSim_4_1}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT/platforms"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export DISPLAY="${DISPLAY:-:0}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${STARVLA_DIR}:${STARVLA_DIR}/examples/simBenchmarks/RLBench/eval_files"

"${RLBENCH_PYTHON}" ./examples/simBenchmarks/RLBench/eval_files/eval_rlbench.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --tasks "${TASKS}" \
  --trials "${TRIALS}" \
  --max-steps "${MAX_STEPS}"