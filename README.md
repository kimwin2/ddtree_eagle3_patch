# Gauss4 DFlash / DDTree Inference Benchmark

## Table of Contents

1. [Quick reference](#quick-reference)
   - [Purpose](#purpose)
   - [Pre-requisites](#pre-requisites)
   - [Configuration](#configuration)
   - [Inference](#inference)
   - [Outputs](#outputs)
2. [Background](#background)
   - [DFlash draft inference](#dflash-draft-inference)
   - [DDTree verification](#ddtree-verification)
   - [Acceptance length measurement](#acceptance-length-measurement)
3. [Repository layout](#repository-layout)

# Quick reference

Home repository for running DFlash and DDTree inference benchmarks on Gauss4 models.

## Purpose

This repository is focused on validating the inference path for:

* Gauss4 target model inference
* DFlash draft model inference
* DDTree verification on top of DFlash draft outputs
* Acceptance length measurement for speculative decoding behavior

The current draft checkpoints are still under training. The generated text quality and benchmark scores should be treated as placeholder values. The primary goal of this branch is to verify that the DFlash and DDTree inference logic runs correctly and that acceptance length / timing metrics can be collected.

## Pre-requisites

This codebase expects a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

The default dependency file targets PyTorch CUDA 12.6 wheels. If your runtime uses a different CUDA stack, install the matching PyTorch build first and then install the remaining dependencies.

Model and draft checkpoint paths are configured in `run_benchmark.sh`.

## Configuration

Edit the following sections in `run_benchmark.sh` before running:

```bash
TASKS=(
  "gsm8k:1"
)

MODEL_DRAFT_PAIRS=(
  "/path/to/gauss4-target|/path/to/dflash-draft-checkpoint"
)

TEMPERATURES=(
  "0.0"
)
```

The first field in each `MODEL_DRAFT_PAIRS` entry is the Gauss4 target model path. The second field is the DFlash draft checkpoint path.

For direct `benchmark.py` usage, the core arguments are:

```bash
torchrun --nproc_per_node=1 benchmark.py \
  --dataset gsm8k \
  --max-samples 1 \
  --model-name-or-path /path/to/gauss4-target \
  --draft-name-or-path /path/to/dflash-draft-checkpoint \
  --tree-budget 31 \
  --max-new-tokens 2048 \
  --temperature 0.0 \
  --save-path runs/sample.pt
```

Useful options:

* `--tree-budget`: DDTree node budget. Multiple values can be passed as comma-separated values.
* `--block-size`: draft block size. If omitted, the draft model config is used.
* `--flash-attn`: runs the DFlash path without DDTree.
* `--draft-type`: `dflash` or `littlebit_dflash`.
* `--apply-ee`: enables the early-exit feature path used by the current Gauss4 draft experiments.

## Inference

Run the benchmark script:

```bash
bash run_benchmark.sh
```

The script writes logs to `logs/` and serialized run outputs to `runs/`.

By default, the benchmark runs:

* baseline autoregressive target generation
* DFlash speculative generation
* DDTree speculative generation for each configured tree budget

When `--flash-attn` is passed, only the DFlash speculative path is run.

## Outputs

Each saved `.pt` file contains a dictionary with:

* `responses`: per-sample outputs for baseline, DFlash, and DDTree runs
* `block_size`: draft block size used for the run
* `draft_algorithm`: selected draft algorithm
* `draft_attn_implementation`: draft attention backend
* `target_attn_implementation`: target attention backend
* `args`: parsed CLI arguments

Each response object includes:

* `output_ids`
* `num_input_tokens`
* `num_output_tokens`
* `time_to_first_token`
* `time_per_output_token`
* `acceptance_lengths`
* `decode_rounds`
* `stage_times`

`acceptance_lengths` is the main signal for checking speculative decoding behavior while the draft model is still being trained.

# Background

## DFlash draft inference

DFlash predicts a block of future tokens with a draft model. The target model then verifies those draft tokens and commits the accepted prefix. This repository keeps the DFlash path available as a baseline speculative decoding mode.

## DDTree verification

DDTree builds a tree from DFlash draft logits and verifies multiple candidate paths with the target model. This can increase the chance of accepting more tokens per verification step compared with a single linear draft block.

The DDTree implementation records stage timing for draft, tree build, tree compile, verify, and commit stages.

## Acceptance length measurement

Acceptance length is the number of tokens committed in one speculative decoding round. Larger acceptance lengths usually indicate that the draft model is closer to the target model distribution and that fewer target verification rounds are needed.

For the current Gauss4 checkpoints, acceptance lengths are intended for logic validation and training progress inspection rather than final performance claims.

# Repository layout

* `benchmark.py`: main benchmark entry point
* `dflash.py`: DFlash speculative generation path
* `ddtree.py`: DDTree speculative generation path
* `model/`: draft model and dataset utilities
* `littlebit/`: quantized DFlash draft loading utilities
* `run_benchmark.sh`: example benchmark launcher
