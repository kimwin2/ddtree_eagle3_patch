"""Web demo: three decoding algorithms streaming side by side.

Spawns one GPU worker per algorithm (baseline autoregressive, DFlash, DFlash +
DDTree), each pinned to its own GPU, and streams committed tokens to the browser
in real time over Server-Sent Events. The page shows the live text, a live
tokens-per-second counter for all three, and the average acceptance length for
the two speculative methods.

Run:
    python -m demo.server \
        --model-name-or-path Qwen/Qwen3-8B \
        --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16 \
        --gpus 0,1,2 \
        --model-label "Gauss 4.0"
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

# Make the repo root importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import multiprocessing as mp

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from demo.worker import worker_main


METHODS = ["baseline", "dflash", "ddtree"]

# Filled in by main().
CONFIG = {}
IN_QUEUES = {}        # method -> mp.Queue of requests
OUT_Q = None          # shared mp.Queue of events from all workers
REGISTRY = {}         # req_id -> thread queue.Queue (one per in-flight browser request)
REGISTRY_LOCK = threading.Lock()
REQ_COUNTER = {"n": 0}
TOKENIZER = None

app = FastAPI()

_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


def _dispatcher_loop():
    """Route worker events to the right in-flight request by req id."""
    while True:
        event = OUT_Q.get()
        if event is None:
            break
        req_id = event.get("req")
        with REGISTRY_LOCK:
            sink = REGISTRY.get(req_id)
        if sink is not None:
            sink.put(event)


@app.get("/")
def index():
    with open(_HTML_PATH, "r", encoding="utf-8") as handle:
        return HTMLResponse(handle.read())


@app.get("/config")
def config():
    rom = CONFIG.get("rom", {})
    return {
        "methods": [
            {
                "key": key,
                "title": CONFIG["titles"][key],
                "target_rom_bytes": rom.get(key, {}).get("target_rom_bytes", 0),
                "draft_rom_bytes": rom.get(key, {}).get("draft_rom_bytes", 0),
            }
            for key in METHODS
        ],
        "default_max_new_tokens": CONFIG["max_new_tokens"],
        "default_temperature": CONFIG["temperature"],
    }


def _new_req_id() -> int:
    with REGISTRY_LOCK:
        REQ_COUNTER["n"] += 1
        return REQ_COUNTER["n"]


@app.get("/stream")
def stream(request: Request, prompt: str, max_new_tokens: int = None, temperature: float = None):
    max_new_tokens = max_new_tokens or CONFIG["max_new_tokens"]
    temperature = CONFIG["temperature"] if temperature is None else temperature

    req_id = _new_req_id()
    sink: "queue.Queue" = queue.Queue()
    with REGISTRY_LOCK:
        REGISTRY[req_id] = sink

    # Each column renders its OWN committed tokens independently. At temperature 0
    # the three methods are usually identical anyway (modulo tiny float argmax
    # drift), which now shows up honestly per column instead of being unified.
    state = {
        key: {"text": "", "count": 0, "t0": None, "acc_sum": 0.0, "acc_rounds": 0, "ids": []}
        for key in METHODS
    }
    errored = set()

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def event_stream():
        try:
            yield sse({"event": "start", "req": req_id})
            for key in METHODS:
                IN_QUEUES[key].put(
                    {
                        "req": req_id,
                        "prompt": prompt,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    }
                )

            finished = set()
            while len(finished) < len(METHODS):
                try:
                    event = sink.get(timeout=0.5)
                except queue.Empty:
                    # keep the connection alive
                    yield ": keep-alive\n\n"
                    continue

                method = event["method"]
                st = state[method]

                if event["type"] == "token":
                    if st["t0"] is None:
                        st["t0"] = time.time()
                    st["ids"].extend(event["ids"])
                    st["count"] = len(st["ids"])
                    st["acc_sum"] += float(event["acc"])
                    st["acc_rounds"] += 1
                    full = TOKENIZER.decode(st["ids"], skip_special_tokens=True)
                    delta = full[len(st["text"]) :]
                    st["text"] = full
                    elapsed = max(time.time() - st["t0"], 1e-6)
                    live_tps = st["count"] / elapsed
                    live_acc = st["acc_sum"] / max(st["acc_rounds"], 1)
                    yield sse(
                        {
                            "event": "token",
                            "method": method,
                            "delta": delta,
                            "tps": round(live_tps, 1),
                            "acc": round(live_acc, 2),
                            "tokens": st["count"],
                            "ram_bytes": event.get("ram_bytes", 0),
                        }
                    )
                elif event["type"] == "done":
                    finished.add(method)
                    yield sse(
                        {
                            "event": "done",
                            "method": method,
                            "tps": round(event["tps"], 1),
                            "acc": round(event["acc"], 2),
                            "tokens": event["num_tokens"],
                            "ttft": round(event["ttft"] * 1000),
                            "ram_bytes": event.get("ram_bytes", 0),
                        }
                    )
                elif event["type"] == "error":
                    finished.add(method)
                    errored.add(method)
                    yield sse({"event": "error", "method": method, "error": event["error"]})

            # Send each (non-errored) column its OWN final text.
            yield sse({
                "event": "final",
                "texts": {key: state[key]["text"] for key in METHODS},
                "errored": sorted(errored),
            })
        finally:
            with REGISTRY_LOCK:
                REGISTRY.pop(req_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2",
        help="Comma-separated physical GPU ids for baseline,dflash,ddtree (in that order).",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default="Gemma4-E2B",
        help="Display name for the target model; column titles are built from it.",
    )
    parser.add_argument("--tree-budget", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if len(gpus) < len(METHODS):
        raise SystemExit(f"Need {len(METHODS)} GPU ids (got {gpus}); e.g. --gpus 0,1,2")

    model_label = args.model_label or os.path.basename(args.model_name_or_path.rstrip("/"))

    global CONFIG, OUT_Q, TOKENIZER
    CONFIG = {
        "titles": {
            "baseline": model_label,
            "dflash": f"{model_label} + DFlash",
            "ddtree": f"{model_label} + DFlash + DDTree",
        },
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "rom": {},  # method -> {target_rom_bytes, draft_rom_bytes}, filled as workers report ready.
    }

    # Server-side tokenizer for incremental decoding (CPU only, no GPU needed).
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(args.model_name_or_path)

    ctx = mp.get_context("spawn")
    OUT_Q = ctx.Queue()
    ready_q = ctx.Queue()

    procs = []
    for method, gpu in zip(METHODS, gpus):
        in_q = ctx.Queue()
        IN_QUEUES[method] = in_q
        proc = ctx.Process(
            target=worker_main,
            args=(
                method,
                gpu,
                args.model_name_or_path,
                args.draft_name_or_path,
                args.tree_budget,
                in_q,
                OUT_Q,
                ready_q,
            ),
            daemon=True,
        )
        proc.start()
        procs.append(proc)
        print(f"[demo] launched worker method={method} on GPU {gpu} (pid={proc.pid})", flush=True)

    # Wait for all workers to finish loading + warming up.
    ready = 0
    while ready < len(METHODS):
        msg = ready_q.get()
        if isinstance(msg, dict) and msg.get("error"):
            raise SystemExit(f"[demo] worker '{msg.get('method')}' failed to start:\n{msg['error']}")
        CONFIG["rom"][msg["method"]] = {
            "target_rom_bytes": int(msg.get("target_rom_bytes", 0)),
            "draft_rom_bytes": int(msg.get("draft_rom_bytes", 0)),
        }
        print(f"[demo] worker ready: {msg}", flush=True)
        ready += 1

    dispatcher = threading.Thread(target=_dispatcher_loop, daemon=True)
    dispatcher.start()

    print(f"[demo] all workers ready — serving on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
