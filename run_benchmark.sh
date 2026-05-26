#!/usr/bin/env bash

set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29600}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DIR="${RUN_DIR:-runs}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

TASKS=(
  "gsm8k:1"
)

MODEL_DRAFT_PAIRS=(
  "/group-volume/models/gemma-4-E2B-it|/group-volume/bs93.lee/LittleD/checkpoints/gemma4-e2b-dflash-bf16-8gpu-v7-swa-h/epoch_6_step_268890"
)

TEMPERATURES=(
  "0.0"
)

COMMON_BENCHMARK_ARGS=(
  --max-new-tokens 2048
)

DRAFT_CONFIGS=(
  "dflash|||||||"
)

slugify() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

build_draft_args() {
  local config="$1"
  IFS='|' read -r draft_type quant_mod quant_func eff_bit kv_factor min_split_dim group_size residual <<< "${config}"

  local args=()
  args+=(--draft-type "${draft_type}")

  if [[ "${draft_type}" == "littlebit_dflash" ]]; then
    [[ -n "${quant_mod}" ]] && args+=(--quant-mod "${quant_mod}")
    [[ -n "${quant_func}" ]] && args+=(--quant-func "${quant_func}")
    [[ -n "${eff_bit}" ]] && args+=(--eff-bit "${eff_bit}")
    [[ -n "${kv_factor}" ]] && args+=(--kv-factor "${kv_factor}")
    [[ -n "${min_split_dim}" ]] && args+=(--min-split-dim "${min_split_dim}")
    [[ -n "${group_size}" ]] && args+=(--group-size "${group_size}")
    [[ "${residual}" == "true" ]] && args+=(--residual)
  fi

  echo "${args[*]}"
}

run_benchmark() {
  local dataset_name="$1"
  local max_samples="$2"
  local model_name="$3"
  local draft_name="$4"
  local mode_name="$5"
  local save_path="$6"
  local log_path="$7"
  shift 7

  echo "========================================================"
  echo "Running Benchmark: dataset=${dataset_name} max_samples=${max_samples} model=${model_name} draft=${draft_name} mode=${mode_name}"
  echo "========================================================"

  if [[ -f "${save_path}" ]]; then
    echo "Skipping existing run: ${save_path}"
    return
  fi

  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    benchmark.py \
    --dataset "${dataset_name}" \
    --max-samples "${max_samples}" \
    --model-name-or-path "${model_name}" \
    --draft-name-or-path "${draft_name}" \
    --save-path "${save_path}" \
    "${COMMON_BENCHMARK_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${log_path}"
}

for task in "${TASKS[@]}"; do
  IFS=':' read -r dataset_name max_samples <<< "${task}"

  for pair in "${MODEL_DRAFT_PAIRS[@]}"; do
    IFS='|' read -r model_name draft_name <<< "${pair}"

    model_slug="$(slugify "${model_name}")"
    draft_slug="$(slugify "${draft_name}")"
    for temperature in "${TEMPERATURES[@]}"; do
      temperature_slug="$(slugify "${temperature}")"
      run_name="${dataset_name}__${model_slug}__${draft_slug}__temp${temperature_slug}"

      for draft_config in "${DRAFT_CONFIGS[@]}"; do
        draft_args_str="$(build_draft_args "${draft_config}")"
        read -ra draft_args <<< "${draft_args_str}"

        run_benchmark \
          "${dataset_name}" \
          "${max_samples}" \
          "${model_name}" \
          "${draft_name}" \
          "sdpa" \
          "${RUN_DIR}/${run_name}__sdpa.pt" \
          "${LOG_DIR}/${run_name}__sdpa.log" \
          --temperature "${temperature}" \
          "${draft_args[@]}"

      done
    done
  done
done
