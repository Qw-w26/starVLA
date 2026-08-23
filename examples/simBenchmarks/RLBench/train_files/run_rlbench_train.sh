#!/usr/bin/env bash
# RLBench — launch script (smoke-test profile by default).
# Run from the repo root, e.g.:
#   bash examples/simBenchmarks/RLBench/train_files/run_rlbench_train.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # always run from repo root

###########################################################################################
# === Edit these for your environment ===
Framework_name=QwenPI_v3          # QwenOFT | QwenFAST | QwenPI | QwenPI_v3 | QwenGR00T
freeze_module_list=''             # e.g. 'qwen_vl' to freeze the VLM backbone
config_yaml=./examples/simBenchmarks/RLBench/train_files/starvla_cotrain_rlbench.yaml
data_root_dir=./playground/Datasets/RLBENCH
data_mix=rlbench_all
run_root_dir=./playground/Checkpoints
run_id=starvla_qwenpi_rlbench_smoke
num_processes=2                   # number of GPUs
###########################################################################################

mkdir -p "${run_root_dir}/${run_id}"
cp "$0" "${run_root_dir}/${run_id}/$(basename "$0")"          # snapshot the launcher
cp "${config_yaml}" "${run_root_dir}/${run_id}/$(basename "${config_yaml}")"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name "${Framework_name}" \
    --datasets.vla_data.data_root_dir "${data_root_dir}" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    ${freeze_module_list:+--trainer.freeze_modules "${freeze_module_list}"}