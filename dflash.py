import os
import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import (
    DFlashDraftModel,
    apply_logit_processing,
    compute_target_lm_logits,
    embed_target_input_ids,
    sample,
    extract_context_feature,
    get_model_text_config,
)


DFLASH_STAGE_ORDER = ("draft", "verify", "commit")

_DFLASH_DEBUG_STATE = {"count": 0}


def build_dflash_target_attention_mask(
    target: AutoModelForCausalLM,
    past_length: int,
    query_length: int,
    query_position_ids: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | dict[str, torch.Tensor] | None:
    """Build a sliding-window-aware attention mask for a target chain forward.

    For hybrid-attention models (e.g. Gemma4) the sliding layers must only see
    the last ``sliding_window`` keys, but the transformers default mask builder
    is not guaranteed to apply this restriction when the cache is a plain
    DynamicCache. Building the mask explicitly avoids the silent failure where
    sequence length exceeds the sliding window and sliding layers degrade into
    full attention.
    """
    target_config = get_model_text_config(target)
    layer_types = getattr(target_config, "layer_types", None)
    if not layer_types or "sliding_attention" not in layer_types:
        return None
    sliding_window = getattr(target_config, "sliding_window", None)
    if sliding_window is None or sliding_window <= 0:
        return None

    kv_length = past_length + query_length
    key_positions = torch.arange(kv_length, device=device, dtype=query_position_ids.dtype)
    query_positions = query_position_ids[0]

    causal_visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    sliding_visible = causal_visible & (
        key_positions.unsqueeze(0) > (query_positions.unsqueeze(1) - int(sliding_window))
    )

    minval = torch.finfo(dtype).min
    full_mask = torch.zeros((1, 1, query_length, kv_length), dtype=dtype, device=device)
    full_mask.masked_fill_(~causal_visible[None, None, :, :], minval)
    sliding_mask = torch.zeros_like(full_mask)
    sliding_mask.masked_fill_(~sliding_visible[None, None, :, :], minval)
    return {"full_attention": full_mask, "sliding_attention": sliding_mask}


def dflash_debug_dump_target_hidden(tag, target_hidden, target_layer_ids=None):
    """Dump the raw target context feature for cross-framework comparison.

    Enabled by setting the DFLASH_DEBUG_DUMP env var to an output directory.
    Dumps at most DFLASH_DEBUG_LIMIT (default 4) tensors so that a single
    request can be inspected and compared against the vLLM reference.
    """
    dump_dir = os.environ.get("DFLASH_DEBUG_DUMP")
    if not dump_dir:
        return
    idx = _DFLASH_DEBUG_STATE["count"]
    if idx >= int(os.environ.get("DFLASH_DEBUG_LIMIT", "4")):
        return
    _DFLASH_DEBUG_STATE["count"] = idx + 1
    tensor = target_hidden.detach()
    tensor_f = tensor.float()
    summary = {
        "framework": "hf",
        "tag": tag,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "target_layer_ids": list(target_layer_ids) if target_layer_ids is not None else None,
        "mean": tensor_f.mean().item(),
        "std": tensor_f.std().item(),
        "abs_mean": tensor_f.abs().mean().item(),
        "per_token_norm_head": tensor_f.norm(dim=-1).flatten()[:8].tolist(),
    }
    print(f"[DFLASH-DEBUG] {summary}", flush=True)
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"hf_target_hidden_{idx:03d}_{tag}.pt")
    torch.save({"summary": summary, "target_hidden": tensor.cpu()}, path)
    print(f"[DFLASH-DEBUG] saved {path}", flush=True)


def format_top_logits(logits: torch.Tensor, k: int = 5) -> str:
    top_values, top_indices = torch.topk(logits.float(), k=min(k, logits.shape[-1]), dim=-1)
    return ",".join(
        f"{int(token_id.item())}:{float(value.item()):.6f}"
        for token_id, value in zip(top_indices, top_values)
    )


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    debug_expected_output_ids: torch.Tensor | None = None,
    debug_label: str = "",
    debug_mismatch_log_limit: int | None = 8,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DFLASH_STAGE_ORDER)

    prefill_start = cuda_time()
    prefill_attention_mask = build_dflash_target_attention_mask(
        target=target,
        past_length=0,
        query_length=num_input_tokens,
        query_position_ids=position_ids[:, :num_input_tokens],
        dtype=target.dtype,
        device=target.device,
    )
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
        attention_mask=prefill_attention_mask,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        dflash_debug_dump_target_hidden("prefill", target_hidden, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    draft_prefill = True
    debug_mismatch_count = 0
    debug_mismatch_suppressed = False

    def emit_verify_mismatch(message: str) -> None:
        nonlocal debug_mismatch_count, debug_mismatch_suppressed
        if debug_mismatch_log_limit is not None and debug_mismatch_log_limit <= 0:
            return
        if debug_mismatch_log_limit is None or debug_mismatch_count < debug_mismatch_log_limit:
            print(message, flush=True)
            debug_mismatch_count += 1
            return
        if not debug_mismatch_suppressed:
            print(
                f"[DFLASH-VERIFY-MISMATCH-SUPPRESSED] {debug_label} "
                f"further mismatch logs suppressed after {debug_mismatch_log_limit} messages",
                flush=True,
            )
            debug_mismatch_suppressed = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            draft_stage_start = cuda_time()
            noise_embedding = embed_target_input_ids(target, block_output_ids)
            draft_logits = compute_target_lm_logits(target, model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1 :, :])
            draft_logits = apply_logit_processing(draft_logits, model.logit_scale, model.final_logit_softcapping)
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            draft_stage_elapsed = cuda_time() - draft_stage_start
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()
            else:
                stage_times["draft"] += draft_stage_elapsed

        verify_stage_start = cuda_time()
        verify_attention_mask = build_dflash_target_attention_mask(
            target=target,
            past_length=start,
            query_length=block_size,
            query_position_ids=block_position_ids,
            dtype=target.dtype,
            device=target.device,
        )
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
            attention_mask=verify_attention_mask,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        if debug_expected_output_ids is not None:
            expected_ids = debug_expected_output_ids.to(device=posterior.device)
            for accepted_offset in range(1, acceptance_length + 1):
                commits_abs = start + accepted_offset
                if commits_abs >= expected_ids.shape[1]:
                    break
                expected = expected_ids[0, commits_abs]
                actual = block_output_ids[0, accepted_offset]
                if bool((actual != expected).item()):
                    logit_idx = accepted_offset - 1
                    emit_verify_mismatch(
                        f"[DFLASH-VERIFY-MISMATCH] {debug_label} "
                        f"round_start_abs={start} round_start_gen={start - num_input_tokens} "
                        f"logit_idx={logit_idx} commits_abs={commits_abs} "
                        f"commits_gen={commits_abs - num_input_tokens} "
                        f"expected={int(expected.item())} committed={int(actual.item())} "
                        f"expected_logit={float(output.logits[0, logit_idx, expected].float().item()):.6f} "
                        f"committed_logit={float(output.logits[0, logit_idx, actual].float().item()):.6f} "
                        f"top_logits={format_top_logits(output.logits[0, logit_idx])}"
                    )
                    break
            else:
                predicts_abs = start + acceptance_length + 1
                if predicts_abs < expected_ids.shape[1]:
                    expected = expected_ids[0, predicts_abs]
                    actual = posterior[0, acceptance_length]
                    if bool((actual != expected).item()):
                        emit_verify_mismatch(
                            f"[DFLASH-VERIFY-MISMATCH] {debug_label} "
                            f"round_start_abs={start} round_start_gen={start - num_input_tokens} "
                            f"logit_idx={acceptance_length} predicts_abs={predicts_abs} "
                            f"predicts_gen={predicts_abs - num_input_tokens} "
                            f"expected={int(expected.item())} posterior={int(actual.item())} "
                            f"expected_logit={float(output.logits[0, acceptance_length, expected].float().item()):.6f} "
                            f"posterior_logit={float(output.logits[0, acceptance_length, actual].float().item()):.6f} "
                            f"top_logits={format_top_logits(output.logits[0, acceptance_length])}"
                        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]
            dflash_debug_dump_target_hidden(f"round{len(acceptance_lengths)}", target_hidden, model.target_layer_ids)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - acceptance_length - 1 : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
    )


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def empty_stage_times(stage_names: tuple[str, ...]) -> dict[str, float]:
    return {stage_name: 0.0 for stage_name in stage_names}
