# Speculative Decoding Demo

A three-column live demo that streams the **same prompt** through three decoders at
once, each on its own GPU, so you can watch the throughput difference in real time:

| Column | Algorithm | What it shows |
|--------|-----------|---------------|
| 1 | Autoregressive (baseline) | original target model, one token per step |
| 2 | DFlash | block speculative decoding, draft + target |
| 3 | DFlash + DDTree | tree verification on top of DFlash |

All three stream characters as they are committed, with a live **tokens/sec**
counter. Columns 2 and 3 also show the live **acceptance length**. The decoding
font is intentionally small so the speed gap is easy to feel.

## How it works

- `worker.py` — one subprocess per algorithm, pinned to one GPU via
  `CUDA_VISIBLE_DEVICES`. Loads the target (and DFlash draft) once, then serves
  generation requests, streaming committed tokens back through a callback.
- `server.py` — FastAPI app. Spawns the three workers, fans each prompt out to all
  of them, and streams merged token events to the browser over Server-Sent Events.
  Incremental detokenization happens server-side (CPU).
- `index.html` — the UI (prompt box + three live columns).

The token streaming is enabled by an optional `on_commit` callback that was added
to `target_generate` / `dflash_generate` / `ddtree_generate` (no behavior change
when the callback is omitted, so the benchmark path is untouched).

## Install

```bash
pip install -r requirements.txt          # repo root deps
pip install -r demo/requirements.txt     # fastapi + uvicorn
```

## Run (one shot)

```bash
bash demo/run_demo.sh
```

Everything is overridable via env vars:

```bash
GPUS=2,3,4 MODEL="Qwen/Qwen3-4B" DRAFT="z-lab/Qwen3-4B-DFlash-b16" \
  MODEL_LABEL="Gauss 4.0" PORT=9000 bash demo/run_demo.sh
```

`run_demo.sh` installs `demo/requirements.txt` (skip with `INSTALL_DEPS=0`) and
then launches the server below.

## Run (manual, 3 GPUs)

```bash
python -m demo.server \
  --model-name-or-path Qwen/Qwen3-8B \
  --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16 \
  --gpus 0,1,2 \
  --model-label "Gauss 4.0" \
  --host 0.0.0.0 --port 8000
```

Then open `http://<server-ip>:8000`, type a prompt, and hit **Run**
(or Ctrl/Cmd+Enter).

### Options

| Flag | Default | Notes |
|------|---------|-------|
| `--model-name-or-path` | `Qwen/Qwen3-8B` | target model (shared by all three) |
| `--draft-name-or-path` | `z-lab/Qwen3-8B-DFlash-b16` | DFlash draft |
| `--gpus` | `0,1,2` | one GPU per column: baseline, dflash, ddtree |
| `--model-label` | model basename | the name shown in each column header |
| `--tree-budget` | `256` | DDTree tree node budget |
| `--max-new-tokens` | `512` | per-request generation cap |
| `--temperature` | `0.0` | greedy by default; at 0 the three outputs match |

> The target verifier runs with `sdpa` attention in all three workers because
> DDTree uses a custom tree attention mask that is incompatible with
> FlashAttention. The DFlash draft uses FlashAttention when `flash_attn` is
> installed, otherwise it falls back to `sdpa`.
