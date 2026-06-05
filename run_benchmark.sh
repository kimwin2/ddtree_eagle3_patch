#!/usr/bin/env bash

set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29600}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DIR="${RUN_DIR:-runs}"
EAGLE3_BATCH_SIZE="${EAGLE3_BATCH_SIZE:-1}"
EAGLE3_DEPTH="${EAGLE3_DEPTH:-7}"
EAGLE3_TOPK="${EAGLE3_TOPK:-8}"
EAGLE3_TREE_SIZE="${EAGLE3_TREE_SIZE:-32}"

# DFlash draft uint16 activation quantization (A16 PTQ, static). Calibration uses
# the HELD-OUT samples of the eval datasets (those not consumed by eval).
# Set ENABLE_DRAFT_QUANT=0 to run the original (non-quantized) benchmark.
ENABLE_DRAFT_QUANT="${ENABLE_DRAFT_QUANT:-1}"
QUANT_OBSERVER="${QUANT_OBSERVER:-minmax}"
QUANT_NUM_BITS="${QUANT_NUM_BITS:-16}"
CALIB_NUM_SAMPLES="${CALIB_NUM_SAMPLES:-128}"
CALIB_SEQ_LEN="${CALIB_SEQ_LEN:-2048}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

TASKS=(
  "gsm8k:128"
  "math500:128"
  "aime24:30"
  "aime25:30"
  "humaneval:164"
  "mbpp:128"
  "livecodebench:128"
  "swe-bench:128"
  "mt-bench:80"
  "alpaca:128"
)

MODEL_DRAFT_PAIRS=(
  "Qwen/Qwen3-4B|z-lab/Qwen3-4B-DFlash-b16"
  "Qwen/Qwen3-8B|z-lab/Qwen3-8B-DFlash-b16"
  "Qwen/Qwen3-Coder-30B-A3B-Instruct|z-lab/Qwen3-Coder-30B-A3B-DFlash"
)

TEMPERATURES=(
  "0.0"
  "1.0"
)

# Calibration spec = the eval tasks; the reader pools each dataset's held-out
# (not-used-by-eval) samples. Built by joining TASKS with commas, e.g.
# "gsm8k:128,math500:128,...". Keeping it identical to TASKS guarantees no
# calibration/eval data leakage.
CALIB_TASKS_SPEC="$(IFS=,; echo "${TASKS[*]}")"

COMMON_BENCHMARK_ARGS=(
  --max-new-tokens 2048
)

slugify() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
}

is_eagle3_draft() {
  local draft_name="$1"
  local lower="${draft_name,,}"
  [[ "${lower}" == *"eagle3"* ]]
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

    # Calibrated qparams are independent of dataset/temperature, so calibrate
    # once per (target, draft) pair and reuse via this cache across all runs.
    quant_cache_path="${RUN_DIR}/qparams__${model_slug}__${draft_slug}__${QUANT_OBSERVER}_b${QUANT_NUM_BITS}.pt"
    QUANT_ARGS=()
    if [[ "${ENABLE_DRAFT_QUANT}" == "1" ]]; then
      QUANT_ARGS=(
        --draft-activation-quant
        --quant-observer "${QUANT_OBSERVER}"
        --quant-num-bits "${QUANT_NUM_BITS}"
        --calib-num-samples "${CALIB_NUM_SAMPLES}"
        --calib-seq-len "${CALIB_SEQ_LEN}"
        --calib-holdout-tasks "${CALIB_TASKS_SPEC}"
        --quant-cache-path "${quant_cache_path}"
      )
    fi

    for temperature in "${TEMPERATURES[@]}"; do
      temperature_slug="$(slugify "${temperature}")"
      run_name="${dataset_name}__${model_slug}__${draft_slug}__temp${temperature_slug}"

      if is_eagle3_draft "${draft_name}"; then
        eagle3_mode="eagle3_b${EAGLE3_BATCH_SIZE}_d${EAGLE3_DEPTH}_k${EAGLE3_TOPK}_t${EAGLE3_TREE_SIZE}"
        run_benchmark \
          "${dataset_name}" \
          "${max_samples}" \
          "${model_name}" \
          "${draft_name}" \
          "${eagle3_mode}" \
          "${RUN_DIR}/${run_name}__${eagle3_mode}.pt" \
          "${LOG_DIR}/${run_name}__${eagle3_mode}.log" \
          --temperature "${temperature}" \
          --draft-algorithm eagle3 \
          --eagle3-batch-size "${EAGLE3_BATCH_SIZE}" \
          --eagle3-depth "${EAGLE3_DEPTH}" \
          --eagle3-topk "${EAGLE3_TOPK}" \
          --eagle3-tree-size "${EAGLE3_TREE_SIZE}"
      else
        run_benchmark \
          "${dataset_name}" \
          "${max_samples}" \
          "${model_name}" \
          "${draft_name}" \
          "sdpa" \
          "${RUN_DIR}/${run_name}__sdpa.pt" \
          "${LOG_DIR}/${run_name}__sdpa.log" \
          --temperature "${temperature}" \
          "${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}"

        run_benchmark \
          "${dataset_name}" \
          "${max_samples}" \
          "${model_name}" \
          "${draft_name}" \
          "flash_attn" \
          "${RUN_DIR}/${run_name}__flash_attn.pt" \
          "${LOG_DIR}/${run_name}__flash_attn.log" \
          --temperature "${temperature}" \
          --flash-attn \
          "${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}"
      fi
    done
  done
done
