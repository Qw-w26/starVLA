#!/usr/bin/env bash
# Start the starVLA inference server for an RLBench-trained checkpoint.
# Run this from the server machine (starVLA env, GPU). Terminal 1.
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/rlbench_example/checkpoints/steps_10000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
)
if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

echo "[rlbench-server] ckpt=${CKPT} gpu=${GPU_ID} port=${PORT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"