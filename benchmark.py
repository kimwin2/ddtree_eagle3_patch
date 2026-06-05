import argparse
import random
from itertools import chain
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import (
    DFlashDraftModel,
    Eagle3DraftModel,
    load_and_process_dataset,
    CalibrationDataReader,
    calibrate_dflash_activations,
    export_qparams,
    format_quant_summary,
    load_qparams,
)
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from eagle3 import eagle3_generate, target_generate


def get_stop_token_ids(tokenizer, target) -> list[int]:
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


def _generated_ids(response) -> torch.Tensor:
    return response.output_ids[0, response.num_input_tokens :]


def _first_token_mismatch(lhs: torch.Tensor, rhs: torch.Tensor) -> int | None:
    compare_len = min(lhs.numel(), rhs.numel())
    if compare_len > 0:
        mismatch = (lhs[:compare_len] != rhs[:compare_len]).nonzero(as_tuple=True)[0]
        if mismatch.numel() > 0:
            return int(mismatch[0].item())
    if lhs.numel() != rhs.numel():
        return compare_len
    return None


def _format_token(tokenizer, token_ids: torch.Tensor, offset: int) -> str:
    if offset >= token_ids.numel():
        return "<missing>"
    token_id = int(token_ids[offset].item())
    piece = tokenizer.decode([token_id], skip_special_tokens=False)
    return f"{token_id}:{piece!r}"


def _format_token_window(tokenizer, token_ids: torch.Tensor, center: int, radius: int = 4) -> str:
    start = max(center - radius, 0)
    end = min(center + radius + 1, token_ids.numel())
    return " ".join(f"{offset}={_format_token(tokenizer, token_ids, offset)}" for offset in range(start, end))


def _alignment_hint(lhs: torch.Tensor, rhs: torch.Tensor, offset: int, max_shift: int = 16, window: int = 32) -> str:
    for shift in range(1, max_shift + 1):
        lhs_end = min(lhs.numel(), offset + shift + window)
        rhs_end = min(rhs.numel(), offset + window)
        compare_len = min(lhs_end - offset - shift, rhs_end - offset)
        if compare_len > 0 and torch.equal(lhs[offset + shift : offset + shift + compare_len], rhs[offset : offset + compare_len]):
            return f" alignment=method_matches_baseline_after_deleting_{shift}_baseline_tokens"

        lhs_end = min(lhs.numel(), offset + window)
        rhs_end = min(rhs.numel(), offset + shift + window)
        compare_len = min(lhs_end - offset, rhs_end - offset - shift)
        if compare_len > 0 and torch.equal(lhs[offset : offset + compare_len], rhs[offset + shift : offset + shift + compare_len]):
            return f" alignment=baseline_matches_method_after_deleting_{shift}_method_tokens"
    return ""


def _acceptance_hint(method_response, mismatch_offset: int) -> str:
    acceptance_lengths = getattr(method_response, "acceptance_lengths", None)
    if not acceptance_lengths:
        return ""

    offset = 0
    for round_idx, round_length in enumerate(acceptance_lengths):
        next_offset = offset + int(round_length)
        if mismatch_offset < next_offset:
            nearby = acceptance_lengths[max(round_idx - 3, 0) : round_idx + 4]
            return (
                f" accept_round={round_idx} round_start={offset} "
                f"round_len={int(round_length)} local={mismatch_offset - offset} "
                f"nearby_acceptance={nearby}"
            )
        offset = next_offset
    return f" accept_after_recorded_rounds total_accepted={offset}"


def validate_response_tokens(
    tokenizer,
    idx: int,
    turn_idx: int,
    response: dict,
    fail_on_mismatch: bool,
    mask_token_id: int | None = None,
) -> None:
    baseline_ids = _generated_ids(response["baseline"])
    for method_key, method_response in response.items():
        if method_key == "baseline":
            continue
        method_ids = _generated_ids(method_response)
        mismatch_offset = _first_token_mismatch(baseline_ids, method_ids)
        if mismatch_offset is None:
            print(
                f"[TOKEN-MATCH] idx={idx} turn={turn_idx} method={method_key} "
                f"generated_tokens={method_ids.numel()}",
                flush=True,
            )
            continue

        hint = ""
        if mismatch_offset < baseline_ids.numel() and int(baseline_ids[mismatch_offset].item()) == mask_token_id:
            hint = " hint=baseline_token_is_mask_token_id_check_output_trimming"
        hint += _alignment_hint(baseline_ids, method_ids, mismatch_offset)
        hint += _acceptance_hint(method_response, mismatch_offset)

        message = (
            f"[TOKEN-MISMATCH] idx={idx} turn={turn_idx} method={method_key} "
            f"baseline_len={baseline_ids.numel()} method_len={method_ids.numel()} "
            f"first_diff={mismatch_offset} "
            f"baseline={_format_token(tokenizer, baseline_ids, mismatch_offset)} "
            f"method={_format_token(tokenizer, method_ids, mismatch_offset)}"
            f"{hint}"
        )
        print(message, flush=True)
        print(
            f"[TOKEN-WINDOW] idx={idx} turn={turn_idx} method={method_key} "
            f"baseline_window={_format_token_window(tokenizer, baseline_ids, mismatch_offset)}",
            flush=True,
        )
        print(
            f"[TOKEN-WINDOW] idx={idx} turn={turn_idx} method={method_key} "
            f"method_window={_format_token_window(tokenizer, method_ids, mismatch_offset)}",
            flush=True,
        )
        if fail_on_mismatch:
            raise RuntimeError(message)


def detect_draft_algorithm(draft_name_or_path: str) -> str:
    config = AutoConfig.from_pretrained(draft_name_or_path)
    architectures = [architecture.lower() for architecture in getattr(config, "architectures", [])]
    if any("eagle3" in architecture for architecture in architectures):
        return "eagle3"
    if "eagle3" in draft_name_or_path.lower():
        return "eagle3"
    return "dflash"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument(
        "--target-attn-implementation",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default="auto",
        help="Override the target model attention backend for exactness debugging.",
    )
    parser.add_argument(
        "--draft-attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        help="Override the DFlash draft model attention backend for exactness debugging.",
    )
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--draft-algorithm", choices=["auto", "dflash", "eagle3"], default="auto")
    parser.add_argument("--eagle3-batch-size", type=int, default=1)
    parser.add_argument("--eagle3-depth", type=int, default=7)
    parser.add_argument("--eagle3-topk", type=int, default=8)
    parser.add_argument("--eagle3-tree-size", type=int, default=32)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--validate-exact-match", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    # --- DFlash draft activation quantization (uint16 fake quant, A16 PTQ) --- #
    parser.add_argument(
        "--draft-activation-quant",
        action="store_true",
        help="Apply uint16 asymmetric per-tensor fake activation quantization to the DFlash draft model.",
    )
    parser.add_argument("--quant-num-bits", type=int, default=16, help="Activation quantization bit-width (uint).")
    parser.add_argument(
        "--quant-observer",
        choices=["minmax", "percentile"],
        default="minmax",
        help="Calibration observer: min-max (default) or percentile/histogram for heavy outliers.",
    )
    parser.add_argument("--quant-percentile", type=float, default=99.99, help="Percentile for the percentile observer.")
    parser.add_argument("--quant-num-bins", type=int, default=2048, help="Histogram bins for the percentile observer.")
    parser.add_argument("--calib-num-samples", type=int, default=128, help="Number of calibration sequences.")
    parser.add_argument("--calib-seq-len", type=int, default=2048, help="Deployment context length per calibration sequence.")
    parser.add_argument(
        "--calib-prompt-len",
        type=int,
        default=None,
        help="Prompt tokens kept per calibration sample (default: calib-seq-len // 2).",
    )
    parser.add_argument(
        "--calib-dataset",
        type=str,
        default=None,
        help="Single dataset used for calibration (defaults to --dataset). Ignored if --calib-holdout-tasks is set.",
    )
    parser.add_argument(
        "--calib-holdout-tasks",
        type=str,
        default=None,
        help=(
            "Eval-task spec 'name:max,name:max' (same format as run_benchmark.sh TASKS). "
            "Calibrates on the HELD-OUT samples not used by eval, pooled across these datasets."
        ),
    )
    parser.add_argument(
        "--quant-cache-path",
        type=str,
        default=None,
        help="Path to save/load calibrated activation qparams to skip recalibration.",
    )
    args = parser.parse_args()
    if args.fail_on_mismatch:
        args.validate_exact_match = True

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)
    draft_algorithm = detect_draft_algorithm(args.draft_name_or_path) if args.draft_algorithm == "auto" else args.draft_algorithm

    if draft_algorithm == "eagle3" and args.eagle3_batch_size != 1:
        raise NotImplementedError("The local Eagle3 benchmark path currently supports batch size 1.")

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    if args.target_attn_implementation != "auto":
        target_attn_implementation = args.target_attn_implementation
    if draft_algorithm == "dflash":
        draft_attn_implementation = (
            "flash_attention_2"
            if args.draft_attn_implementation == "auto"
            else args.draft_attn_implementation
        )
        if draft_attn_implementation == "flash_attention_2" and not installed_flash_attn:
            raise RuntimeError(
                "flash_attn must be installed when DFlash draft attention uses FlashAttention"
            )
    else:
        draft_attn_implementation = "pytorch"

    if draft_algorithm == "eagle3" and target_attn_implementation == "flash_attention_2":
        logger.warning("Eagle3 tree verification uses a custom attention mask; forcing the target verifier to torch.sdpa.")
        target_attn_implementation = "sdpa"
    elif target_attn_implementation == "sdpa" and installed_flash_attn:
        logger.warning("DDTree uses a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa.")
    elif target_attn_implementation == "eager":
        logger.warning("Loading the target with eager attention for exactness debugging.")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    method_key_to_tree_budget = {}
    block_size = args.block_size
    if draft_algorithm == "dflash":
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=draft_attn_implementation,
            dtype=torch.bfloat16,
        ).to(device).eval()
        if hasattr(draft_model, "configure_for_target"):
            draft_model.configure_for_target(target)
        block_size = args.block_size if args.block_size is not None else draft_model.block_size
        tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
        methods_to_run = ["dflash"]
        if not args.flash_attn:
            ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
            methods_to_run.extend(ddtree_method_keys)
            method_key_to_tree_budget.update({f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets})
    else:
        draft_model = Eagle3DraftModel.from_pretrained(
            args.draft_name_or_path,
            total_tokens=args.eagle3_tree_size,
            depth=args.eagle3_depth,
            top_k=args.eagle3_topk,
            dtype=torch.bfloat16,
        ).to(device).eval()
        draft_model.tie_target_embeddings(target.get_input_embeddings())
        draft_model.init_tree()
        methods_to_run = ["eagle3"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    stop_token_ids = get_stop_token_ids(tokenizer, target)
    logger.info(f"Using stop_token_ids={stop_token_ids}")
    dataset = load_and_process_dataset(args.dataset)

    if args.draft_activation_quant:
        if draft_algorithm != "dflash":
            raise ValueError("--draft-activation-quant is only supported for the dflash draft model (not eagle3).")
        quant_config = {
            "num_bits": args.quant_num_bits,
            "observer": args.quant_observer,
            "percentile": args.quant_percentile,
            "num_bins": args.quant_num_bins,
        }
        cache_path = Path(args.quant_cache_path) if args.quant_cache_path else None
        if cache_path is not None and cache_path.exists():
            draft_model.configure_activation_quant(quant_config)
            cached = torch.load(cache_path, map_location=device)
            num_loaded = load_qparams(draft_model, cached["qparams"])
            draft_model.set_activation_quant_enabled(True)
            logger.info(f"Loaded {num_loaded} calibrated activation quantizers from {cache_path}")
            if dist.is_main():
                cached_summary = cached.get("summary", {})
                print(
                    f"[CALIB] loaded qparams from cache (no recalibration). "
                    f"datasets={cached_summary.get('dataset_counts', cached_summary.get('datasets'))}. "
                    f"Delete {cache_path} to force recalibration and see fresh min/max.",
                    flush=True,
                )
                print(format_quant_summary(draft_model), flush=True)
        else:
            if args.calib_holdout_tasks:
                logger.info(
                    f"Calibrating dflash activations on HELD-OUT data: tasks={args.calib_holdout_tasks} "
                    f"samples={args.calib_num_samples} seq_len={args.calib_seq_len} "
                    f"observer={args.quant_observer} num_bits={args.quant_num_bits}"
                )
                reader = CalibrationDataReader(
                    tokenizer=tokenizer,
                    eval_tasks=args.calib_holdout_tasks,
                    num_samples=args.calib_num_samples,
                    seq_len=args.calib_seq_len,
                    prompt_len=args.calib_prompt_len,
                    verbose=dist.is_main(),
                )
            else:
                calib_dataset = args.calib_dataset or args.dataset
                logger.info(
                    f"Calibrating dflash activations: dataset={calib_dataset} "
                    f"samples={args.calib_num_samples} seq_len={args.calib_seq_len} "
                    f"observer={args.quant_observer} num_bits={args.quant_num_bits}"
                )
                reader = CalibrationDataReader(
                    tokenizer=tokenizer,
                    dataset_name=calib_dataset,
                    num_samples=args.calib_num_samples,
                    seq_len=args.calib_seq_len,
                    prompt_len=args.calib_prompt_len,
                    verbose=dist.is_main(),
                )
            summary = calibrate_dflash_activations(
                model=draft_model,
                target=target,
                reader=reader,
                block_size=block_size,
                temperature=args.temperature,
                quant_config=quant_config,
                enable_after=True,
                progress=dist.is_main(),
                print_summary=dist.is_main(),
            )
            logger.info(f"Calibrated dflash activations: {summary}")
            if cache_path is not None and dist.is_main():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"qparams": export_qparams(draft_model), "summary": summary, "quant_config": quant_config},
                    cache_path,
                )
                logger.info(f"Saved activation qparams to {cache_path}")

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = target_generate(
        target=target,
        input_ids=warmup_input_ids,
        max_new_tokens=warmup_max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        if method_key == "eagle3":
            _ = eagle3_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                max_new_tokens=warmup_max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
            )
        elif method_key == "dflash":
            _ = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
            )
        else:
            _ = ddtree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                tree_budget=method_key_to_tree_budget[method_key],
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
            )

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            response["baseline"] = target_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
            )
            for method_key in methods_to_run:
                if method_key == "eagle3":
                    response[method_key] = eagle3_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        stop_token_ids=stop_token_ids,
                        temperature=args.temperature,
                    )
                elif method_key == "dflash":
                    response[method_key] = dflash_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=stop_token_ids,
                        temperature=args.temperature,
                        debug_expected_output_ids=response["baseline"].output_ids if args.validate_exact_match else None,
                        debug_label=f"idx={idx} turn={len(messages) - 1} method={method_key}",
                    )
                else:
                    response[method_key] = ddtree_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        tree_budget=method_key_to_tree_budget[method_key],
                        stop_token_ids=stop_token_ids,
                        temperature=args.temperature,
                        debug_expected_output_ids=response["baseline"].output_ids if args.validate_exact_match else None,
                        debug_label=f"idx={idx} turn={len(messages) - 1} method={method_key}",
                    )

            if args.validate_exact_match:
                validate_response_tokens(
                    tokenizer=tokenizer,
                    idx=idx,
                    turn_idx=len(messages) - 1,
                    response=response,
                    fail_on_mismatch=args.fail_on_mismatch,
                    mask_token_id=getattr(draft_model, "mask_token_id", None),
                )

            spec_response = response[methods_to_run[-1]]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    run_data = {
        "responses": responses,
        "block_size": block_size,
        "draft_algorithm": draft_algorithm,
        "eagle3_config": {
            "batch_size": args.eagle3_batch_size,
            "depth": args.eagle3_depth,
            "topk": args.eagle3_topk,
            "tree_size": args.eagle3_tree_size,
        } if draft_algorithm == "eagle3" else None,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "args": vars(args),
    }
    
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)


if __name__ == "__main__":
    main()
