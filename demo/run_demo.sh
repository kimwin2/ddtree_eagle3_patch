#!/usr/bin/env bash
#
# One-shot launcher for the four-algorithm speculative-decoding web demo.
# Run from the repo root:   bash demo/run_demo.sh
#
# Every setting can be overridden via env vars, e.g.:
#   GPUS=2,3,4,5 MODEL="Qwen/Qwen3-4B" DRAFT="z-lab/Qwen3-4B-DFlash-b16" bash demo/run_demo.sh
#   LB_MODEL=... LB_DRAFT=... bash demo/run_demo.sh
#   INSTALL_DEPS=0 PORT=9000 bash demo/run_demo.sh

set -euo pipefail

# Resolve repo root from this script's location so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- configuration (override via env) --------------------------------------
# Standard target + draft drive the first three columns (baseline / dflash / ddtree).
MODEL="${MODEL:-/group-volume/bs93.lee/LittleD/gauss4-e2b-0.1}"
DRAFT="${DRAFT:-/group-volume/bs93.lee/LittleD/checkpoints/gauss4-e2b-dflash-bf16-8gpu-v7-ee-swa-a-2/epoch_6_step_282402}"

# LittleBit (4th column): its own rotated target + the quantized draft checkpoint.
LB_MODEL="${LB_MODEL:-/group-volume/bs93.lee/LittleD/gauss4-e2b-0.1-rotated}"
LB_DRAFT="${LB_DRAFT:-/group-volume/bs93.lee/LittleD/checkpoints/gauss4-e2b-dflash-lb2-lbod-pc-5ep-res-1.0bit-1.2e-4lr-8gpu-g64-ee-swa-a-2-r-h0.3/epoch_4_step_233900}"

GPUS="${GPUS:-0,1,2,3}"                    # baseline,dflash,ddtree,littlebit (in that order)
MODEL_LABEL="${MODEL_LABEL:-Gauss 4.0 E2B}"
TREE_BUDGET="${TREE_BUDGET:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"          # set 0 to skip pip install

# DRAFT_TYPE applies to the first three columns only; the LittleBit column always
# loads its quantized draft via the littlebit path regardless of this.
DRAFT_TYPE="${DRAFT_TYPE:-dflash}"

# ---- LittleBit quantization config (always passed; used by the LittleBit column) ---
QUANT_MOD="${QUANT_MOD:-LittleBitOnDeviceLinearPerChannel}"
QUANT_FUNC="${QUANT_FUNC:-STEBinary}"
EFF_BIT="${EFF_BIT:-1.0}"
KV_FACTOR="${KV_FACTOR:-2.0}"
MIN_SPLIT_DIM="${MIN_SPLIT_DIM:-8}"
GROUP_SIZE="${GROUP_SIZE:-64}"
RESIDUAL="${RESIDUAL:-0}"                  # set 1 to enable

# NOTE: --gpus selects *physical* GPU ids; each worker pins its own GPU via
# CUDA_VISIBLE_DEVICES internally, so we deliberately do NOT set a restrictive
# CUDA_VISIBLE_DEVICES here (that would be re-interpreted and could mismatch).

# ---- dependencies ----------------------------------------------------------
if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "[run_demo] installing demo deps (set INSTALL_DEPS=0 to skip)..."
  python -m pip install -q -r "${SCRIPT_DIR}/requirements.txt"
fi

# ---- launch ----------------------------------------------------------------
echo "========================================================"
echo "[run_demo] model=${MODEL}"
echo "[run_demo] draft=${DRAFT}  draft_type=${DRAFT_TYPE}"
echo "[run_demo] littlebit_model=${LB_MODEL}"
echo "[run_demo] littlebit_draft=${LB_DRAFT}"
echo "[run_demo] quant_mod=${QUANT_MOD}  quant_func=${QUANT_FUNC}"
echo "[run_demo] eff_bit=${EFF_BIT}  kv_factor=${KV_FACTOR}  min_split_dim=${MIN_SPLIT_DIM}  group_size=${GROUP_SIZE}  residual=${RESIDUAL}"
echo "[run_demo] gpus=${GPUS}  label='${MODEL_LABEL}'"
echo "[run_demo] tree_budget=${TREE_BUDGET}  max_new_tokens=${MAX_NEW_TOKENS}  temperature=${TEMPERATURE}"
echo "[run_demo] serving on http://${HOST}:${PORT}"
echo "========================================================"

# Quant args are always passed: the LittleBit (4th) column needs them, and the
# standard columns ignore them.
DRAFT_ARGS=()
DRAFT_ARGS+=(--draft-type "${DRAFT_TYPE}")
DRAFT_ARGS+=(--quant-mod "${QUANT_MOD}")
DRAFT_ARGS+=(--quant-func "${QUANT_FUNC}")
DRAFT_ARGS+=(--eff-bit "${EFF_BIT}")
DRAFT_ARGS+=(--kv-factor "${KV_FACTOR}")
DRAFT_ARGS+=(--min-split-dim "${MIN_SPLIT_DIM}")
DRAFT_ARGS+=(--group-size "${GROUP_SIZE}")
if [[ "${RESIDUAL}" == "1" ]]; then
  DRAFT_ARGS+=(--residual)
fi

exec python -m demo.server \
  --model-name-or-path "${MODEL}" \
  --draft-name-or-path "${DRAFT}" \
  --littlebit-model-name-or-path "${LB_MODEL}" \
  --littlebit-draft-name-or-path "${LB_DRAFT}" \
  "${DRAFT_ARGS[@]}" \
  --gpus "${GPUS}" \
  --model-label "${MODEL_LABEL}" \
  --tree-budget "${TREE_BUDGET}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --host "${HOST}" \
  --port "${PORT}"
