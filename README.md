<h1 align="center">DDTree</h1>

<p align="center">
  Official implementation of <strong>DDTree (Diffusion Draft Tree)</strong> from
  <em>Accelerating Speculative Decoding with Block Diffusion Draft Trees</em>.
</p>

<p align="center">
  Liran Ringel, Yaniv Romano
</p>

<p align="center">
  <a href="https://liranringel.github.io/ddtree/">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2604.12989">📄 Paper</a>
</p>

## Setup

This codebase is intended for a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

## Run Experiments

```bash
bash run_benchmark.sh
```

This produces benchmark outputs in `runs/` and logs in `logs/`.

For Eagle3 draft models, put an Eagle3 pair in `MODEL_DRAFT_PAIRS`; `run_benchmark.sh`
will switch to the Eagle3 path automatically when the draft name contains `eagle3`.

```bash
MODEL_DRAFT_PAIRS=(
  "Qwen/Qwen3-4B|AngelSlim/Qwen3-4B_eagle3"
)
```

The default Eagle3 settings are batch size 1, depth 7, top-k 8, and tree size 32.
They can be overridden either through `benchmark.py` args
(`--eagle3-batch-size`, `--eagle3-depth`, `--eagle3-topk`, `--eagle3-tree-size`)
or through `run_benchmark.sh` environment variables
(`EAGLE3_BATCH_SIZE`, `EAGLE3_DEPTH`, `EAGLE3_TOPK`, `EAGLE3_TREE_SIZE`).

## Reproduce Paper Artifacts

Generate the plots:

```bash
python3 plot_results.py
```

Generate the LaTeX table:

```bash
python3 make_latex_table.py
```

## Citation

```bibtex
@article{ringel2026ddtree,
  title={Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author={Ringel, Liran and Romano, Yaniv},
  journal={arXiv preprint arXiv:2604.12989},
  year={2026}
}
```
