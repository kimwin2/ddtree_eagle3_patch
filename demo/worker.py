"""GPU worker process for the speculative-decoding demo.

Each worker owns ONE GPU and ONE decoding algorithm. It loads the target model
(and the DFlash draft when needed) once, then serves generation requests off an
input queue, streaming committed tokens back over a shared output queue as they
are produced. CUDA_VISIBLE_DEVICES is pinned per-process so the workers run
truly concurrently on separate GPUs.
"""

import os
import sys
import traceback

# Make the repo root importable regardless of how we are launched (spawn re-imports).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def _model_bytes(module) -> int:
    """Static weight footprint of a model: parameters + buffers, in bytes.

    This is the "ROM" figure — the model's parameters turned into memory. We use
    parameter accounting rather than torch.cuda.memory_allocated() so the number
    is exact and independent of allocator padding, and so a low-bit / quantized
    draft honestly reports its smaller footprint.
    """
    if module is None:
        return 0
    total = sum(p.numel() * p.element_size() for p in module.parameters())
    total += sum(b.numel() * b.element_size() for b in module.buffers())
    return int(total)


# Top-level stage keys per method (what sums to the decode time) plus, for
# ddtree, the sub-breakdown of tree_build. The sub keys are NOT added to the
# total — they decompose tree_build, so summing them again would double-count.
_STAGE_LAYOUT = {
    "baseline": (("decode",), {}),
    "dflash": (("draft", "verify", "commit"), {}),
    "ddtree": (
        ("draft", "tree_build", "tree_compile", "verify", "commit"),
        {"tree_build": ("tree_build_copy", "tree_build_heap", "tree_build_visibility")},
    ),
    # littlebit is the ddtree algorithm with a LittleBit-quantized draft; same stages.
    "littlebit": (
        ("draft", "tree_build", "tree_compile", "verify", "commit"),
        {"tree_build": ("tree_build_copy", "tree_build_heap", "tree_build_visibility")},
    ),
}


def _format_stage_times(method, stage_times, rounds, tps, mean_acc) -> str:
    """Render a readable per-stage decode-time breakdown for one request."""
    top_keys, sub = _STAGE_LAYOUT.get(method, (tuple(stage_times.keys()), {}))
    total = sum(float(stage_times.get(key, 0.0)) for key in top_keys)
    lines = [
        f"[stage_times] method={method} rounds={rounds} "
        f"decode={total:.3f}s tps={tps:.1f} acc={mean_acc:.2f}"
    ]
    for key in top_keys:
        elapsed = float(stage_times.get(key, 0.0))
        pct = (100.0 * elapsed / total) if total > 0 else 0.0
        line = f"    {key:<13s}: {elapsed:.4f}s ({pct:5.1f}%)"
        sub_keys = sub.get(key)
        if sub_keys:
            detail = "  ".join(
                f"{sub_key.replace('tree_build_', '')} {float(stage_times.get(sub_key, 0.0)):.4f}"
                for sub_key in sub_keys
            )
            line += f"   [{detail}]"
        lines.append(line)
    return "\n".join(lines)


def _get_stop_token_ids(tokenizer, target) -> list:
    """Same stop-token logic the benchmark uses (inlined to avoid heavy imports)."""
    stop_token_ids = []
    for source in (
        getattr(target, "generation_config", None),
        getattr(target, "config", None),
        getattr(getattr(target, "config", None), "text_config", None),
    ):
        eos_token_id = getattr(source, "eos_token_id", None)
        if eos_token_id is None:
            continue
        if isinstance(eos_token_id, int):
            stop_token_ids.append(eos_token_id)
        else:
            stop_token_ids.extend(int(token_id) for token_id in eos_token_id)
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(int(tokenizer.eos_token_id))
    return sorted(set(stop_token_ids))


def worker_main(method, gpu_id, model_path, draft_path, tree_budget, in_q, out_q, ready_q,
                draft_type="dflash", quant_mod="LittleBitOnDeviceLinearPerChannel",
                quant_func="STEBinary", eff_bit=1.0, kv_factor=2.0,
                min_split_dim=8, group_size=64, residual=False):
    """Entry point for a worker subprocess.

    method:    one of "baseline", "dflash", "ddtree", "littlebit". "littlebit" runs
               the ddtree algorithm with a LittleBit-quantized draft (draft_type
               "littlebit_dflash").
    gpu_id:    physical GPU id this worker pins to (becomes cuda:0 inside the process).
    in_q:      requests come in as dicts {req, prompt, max_new_tokens, temperature}.
    out_q:     shared queue; we push {type, method, req, ...} events.
    ready_q:   we push our method name once models are loaded + warmed up.
    draft_type: "dflash" or "littlebit_dflash".
    quant_mod:  LittleBit quantization module name.
    quant_func: LittleBit quantization function name.
    eff_bit:    effective bit width for quantization.
    kv_factor:  KV cache factor.
    min_split_dim: minimum split dimension.
    group_size: quantization group size.
    residual:   whether to use residual quantization.
    """
    # Must be set before any CUDA context is created.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from model import DFlashDraftModel
        from eagle3 import target_generate
        from dflash import dflash_generate
        from ddtree import ddtree_generate, maybe_enable_cpp_compact

        torch.manual_seed(0)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

        # The target verifier always runs with sdpa: DDTree applies a custom tree
        # attention mask that is incompatible with FlashAttention. Using sdpa for
        # all three keeps the comparison fair.
        target = (
            AutoModelForCausalLM.from_pretrained(
                # model_path, attn_implementation="sdpa", dtype=torch.bfloat16,
                model_path, attn_implementation="sdpa", dtype=torch.float32,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        stop_token_ids = _get_stop_token_ids(tokenizer, target)

        draft = None
        block_size = None
        if method in ("dflash", "ddtree", "littlebit"):
            draft_attn = "sdpa"
            if draft_type == "littlebit_dflash":
                from littlebit import load_quantized_dflash_model
                draft = load_quantized_dflash_model(
                    draft_path,
                    device=device,
                    # torch_dtype=torch.bfloat16,
                    torch_dtype=torch.float32,
                    quant_args=None,  # auto-detect from checkpoint's littlebit_config.json
                    attn_implementation=draft_attn,
                )
            else:
                try:
                    draft = (
                        DFlashDraftModel.from_pretrained(
                            # draft_path, attn_implementation=draft_attn, dtype=torch.bfloat16,
                            draft_path, attn_implementation=draft_attn, dtype=torch.bfloat32,
                            trust_remote_code=True,
                        )
                        .to(device)
                        .eval()
                    )
                except Exception:
                    # Fall back to sdpa if the draft cannot use FlashAttention here.
                    draft = (
                        DFlashDraftModel.from_pretrained(
                            # draft_path, attn_implementation="sdpa", dtype=torch.bfloat16,
                            draft_path, attn_implementation="sdpa", dtype=torch.float32,
                            trust_remote_code=True,
                        )
                        .to(device)
                        .eval()
                    )
            if hasattr(draft, "configure_for_target"):
                draft.configure_for_target(target)
            block_size = draft.block_size

        if method in ("ddtree", "littlebit"):
            maybe_enable_cpp_compact(True)

        def build_input_ids(prompt: str):
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Tokenizer's chat template doesn't take enable_thinking.
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return tokenizer.encode(text, return_tensors="pt").to(device)

        def run(input_ids, max_new_tokens, temperature, on_commit):
            if method == "baseline":
                return target_generate(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    stop_token_ids=stop_token_ids,
                    temperature=temperature,
                    on_commit=on_commit,
                )
            if method == "dflash":
                return dflash_generate(
                    model=draft,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft.mask_token_id,
                    max_new_tokens=max_new_tokens,
                    block_size=block_size,
                    stop_token_ids=stop_token_ids,
                    temperature=temperature,
                    debug_mismatch_log_limit=0,
                    on_commit=on_commit,
                )
            return ddtree_generate(
                model=draft,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft.mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                tree_budget=tree_budget,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
                debug_mismatch_log_limit=0,
                on_commit=on_commit,
            )

        # Static weight footprint ("ROM"). The target is identical across all
        # three methods (shared); only the draft adds to it.
        target_rom_bytes = _model_bytes(target)
        draft_rom_bytes = _model_bytes(draft)

        # Warm up CUDA kernels / caches so the first real request is not penalised.
        warmup_ids = build_input_ids("Hello")
        run(warmup_ids, 8, 0.0, None)

        ready_q.put(
            {
                "method": method,
                "target_rom_bytes": target_rom_bytes,
                "draft_rom_bytes": draft_rom_bytes,
            }
        )

    except Exception:
        ready_q.put({"error": traceback.format_exc(), "method": method})
        return

    # Serve requests until told to stop.
    while True:
        req = in_q.get()
        if req is None:
            break
        req_id = req["req"]
        try:
            input_ids = build_input_ids(req["prompt"])

            # Activation/KV "RAM" peak for this request: high-water mark of
            # allocated memory above the at-rest (weights-only) level. We read the
            # monotonic max_memory_allocated() rather than polling memory_allocated()
            # because the true peak happens transiently inside a CUDA forward call
            # and a Python poll would miss it.
            rest_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            def on_commit(ids, acc, _req_id=req_id, _rest=rest_allocated):
                ram = max(int(torch.cuda.max_memory_allocated()) - _rest, 0)
                out_q.put(
                    {
                        "type": "token",
                        "method": method,
                        "req": _req_id,
                        "ids": ids,
                        "acc": acc,
                        "ram_bytes": ram,
                    }
                )

            response = run(
                input_ids,
                int(req["max_new_tokens"]),
                float(req["temperature"]),
                on_commit,
            )

            generated_ids = response.output_ids[0, response.num_input_tokens :].tolist()
            full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            tpot = response.time_per_output_token
            tps = (1.0 / tpot) if tpot and tpot > 0 else 0.0
            acc_lengths = response.acceptance_lengths or [1]
            mean_acc = sum(acc_lengths) / len(acc_lengths)
            stage_times = {
                key: float(value)
                for key, value in (getattr(response, "stage_times", None) or {}).items()
            }
            # Print a per-stage decode-time breakdown so we can see, e.g., whether
            # ddtree's round is dominated by target verify or by the tree_build /
            # commit (cache-compaction) CPU work — the key question on small models.
            print(
                _format_stage_times(method, stage_times, int(response.decode_rounds), tps, mean_acc),
                flush=True,
            )
            ram_peak = max(int(torch.cuda.max_memory_allocated()) - rest_allocated, 0)
            out_q.put(
                {
                    "type": "done",
                    "method": method,
                    "req": req_id,
                    "text": full_text,
                    "tps": tps,
                    "acc": mean_acc,
                    "num_tokens": int(response.num_output_tokens),
                    "ttft": float(response.time_to_first_token),
                    "rounds": int(response.decode_rounds),
                    "stage_times": stage_times,
                    "ram_bytes": ram_peak,
                }
            )
        except Exception:
            out_q.put(
                {
                    "type": "error",
                    "method": method,
                    "req": req_id,
                    "error": traceback.format_exc(),
                }
            )
