#!/usr/bin/env bash
#
# One-shot launcher for the three-algorithm speculative-decoding web demo.
# Run from the repo root:   bash demo/run_demo.sh
#
# Every setting can be overridden via env vars, e.g.:
#   GPUS=2,3,4 MODEL="Qwen/Qwen3-4B" DRAFT="z-lab/Qwen3-4B-DFlash-b16" bash demo/run_demo.sh
#   INSTALL_DEPS=0 PORT=9000 bash demo/run_demo.sh

set -euo pipefail

# Resolve repo root from this script's location so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- configuration (override via env) --------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT="${DRAFT:-z-lab/Qwen3-8B-DFlash-b16}"
GPUS="${GPUS:-0,1,2}"                      # baseline,dflash,ddtree (in that order)
MODEL_LABEL="${MODEL_LABEL:-Gemma4-E2B}"
TREE_BUDGET="${TREE_BUDGET:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"          # set 0 to skip pip install

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
echo "[run_demo] draft=${DRAFT}"
echo "[run_demo] gpus=${GPUS}  label='${MODEL_LABEL}'"
echo "[run_demo] tree_budget=${TREE_BUDGET}  max_new_tokens=${MAX_NEW_TOKENS}  temperature=${TEMPERATURE}"
echo "[run_demo] serving on http://${HOST}:${PORT}"
echo "========================================================"

exec python -m demo.server \
  --model-name-or-path "${MODEL}" \
  --draft-name-or-path "${DRAFT}" \
  --gpus "${GPUS}" \
  --model-label "${MODEL_LABEL}" \
  --tree-budget "${TREE_BUDGET}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --host "${HOST}" \
  --port "${PORT}"
