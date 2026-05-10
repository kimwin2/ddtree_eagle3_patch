import time
from types import SimpleNamespace
from typing import Callable, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, DynamicCache
from transformers.cache_utils import Cache
from model import (
    DFlashDraftModel,
    apply_final_logit_softcapping,
    sample,
    extract_context_feature,
)


DFLASH_STAGE_ORDER = ("draft", "verify", "commit")
class _FullStorageSlidingWindowCache(DynamicCache):
    def __init__(self, sliding_windows: dict[int, int], num_layers: int) -> None:
        super().__init__()
        self.sliding_windows = sliding_windows
        self.num_layers = num_layers

    @property
    def is_sliding(self):
        return [i in self.sliding_windows for i in range(self.num_layers)]

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        sliding_window = self.sliding_windows.get(layer_idx)
        past_len = self.get_seq_length(layer_idx)

        if sliding_window is None:
            return past_len + query_length, 0

        kv_offset = max(past_len - int(sliding_window) + 1, 0)
        kv_length = past_len + query_length - kv_offset
        return kv_length, kv_offset

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        keys, values = super().update(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )
        sliding_window = self.sliding_windows.get(layer_idx)
        if sliding_window is None:
            return keys, values

        current_length = key_states.shape[-2]
        view_length = min(keys.shape[-2], int(sliding_window) + current_length - 1)
        return keys[..., -view_length:, :], values[..., -view_length:, :]

def _make_target_cache(target: nn.Module) -> DynamicCache:
    layers = getattr(getattr(target, "model", None), "layers", None)
    if layers is None:
        return DynamicCache()

    sliding_windows: dict[int, int] = {}
    for layer_idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        if not getattr(self_attn, "is_sliding", False):
            continue
        sliding_window = getattr(self_attn, "sliding_window", None)
        if sliding_window is not None:
            sliding_windows[layer_idx] = int(sliding_window)

    if not sliding_windows:
        return DynamicCache()
    return _FullStorageSlidingWindowCache(sliding_windows, num_layers=len(layers))


def _first_sliding_layer(target: nn.Module) -> tuple[int, int] | None:
    layers = getattr(getattr(target, "model", None), "layers", None)
    if layers is None:
        return None

    for layer_idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        if not getattr(self_attn, "is_sliding", False):
            continue
        sliding_window = getattr(self_attn, "sliding_window", None)
        if sliding_window is not None:
            return layer_idx, int(sliding_window)
    return None


def _causal_position_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> torch.Tensor:
    disallowed = key_positions[None, :] > query_positions[:, None]
    if sliding_window is not None:
        disallowed |= key_positions[None, :] < (query_positions[:, None] - int(sliding_window) + 1)

    mask = torch.zeros(
        (1, 1, query_positions.numel(), key_positions.numel()),
        dtype=dtype,
        device=query_positions.device,
    )
    mask[0, 0].masked_fill_(disallowed, torch.finfo(dtype).min)
    return mask


def make_linear_attention_mask_for_target(
    target: nn.Module,
    past_key_values: DynamicCache,
    query_position_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor | dict[str, torch.Tensor] | None:
    query_positions = query_position_ids[0].to(device=query_position_ids.device, dtype=torch.long)
    query_length = int(query_positions.numel())
    past_length = past_key_values.get_seq_length()
    full_key_positions = torch.arange(past_length + query_length, device=query_positions.device)
    full_attention_mask = _causal_position_mask(query_positions, full_key_positions, dtype)

    sliding_layer = _first_sliding_layer(target)
    if sliding_layer is None:
        return full_attention_mask

    sliding_layer_idx, sliding_window = sliding_layer
    kv_length, kv_offset = past_key_values.get_mask_sizes(query_length, sliding_layer_idx)
    sliding_key_positions = torch.arange(kv_offset, kv_offset + kv_length, device=query_positions.device)
    sliding_attention_mask = _causal_position_mask(
        query_positions,
        sliding_key_positions,
        dtype,
        sliding_window=sliding_window,
    )
    return {
        "full_attention": full_attention_mask,
        "sliding_attention": sliding_attention_mask,
    }


def _crop_shared_layer_tensor(cache_tensor: torch.Tensor, keep_length: int, total_length: int) -> torch.Tensor:
    seq_length = cache_tensor.shape[-2]
    if seq_length == 0:
        return cache_tensor

    kv_offset = max(total_length - seq_length, 0)
    keep_count = max(min(keep_length, total_length) - kv_offset, 0)
    return cache_tensor.narrow(-2, 0, keep_count)


def _crop_shared_layers(past_key_values: DynamicCache, keep_length: int, total_length: int) -> None:
    shared_layers = getattr(past_key_values, "shared_layers", None)
    if not shared_layers:
        return

    past_key_values.shared_layers = {
        layer_idx: (
            _crop_shared_layer_tensor(key_cache, keep_length, total_length),
            _crop_shared_layer_tensor(value_cache, keep_length, total_length),
        )
        for layer_idx, (key_cache, value_cache) in shared_layers.items()
    }


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
    debug_label: str | None = None,
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

    # past_key_values_target = DynamicCache()
    past_key_values_target = _make_target_cache(target)
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DFLASH_STAGE_ORDER)
    debug_expected_output_ids = (
        None if debug_expected_output_ids is None else debug_expected_output_ids.to(target.device)
    )
    debug_reported_verify_mismatch = False

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        cache_position=position_ids[0, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            draft_stage_start = cuda_time()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1 :, :])
            draft_logits = apply_final_logit_softcapping(draft_logits, model.final_logit_softcapping)
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            draft_stage_elapsed = cuda_time() - draft_stage_start
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()
            else:
                stage_times["draft"] += draft_stage_elapsed

        verify_stage_start = cuda_time()
        attention_mask = None
        if block_size > 1:
            attention_mask = make_linear_attention_mask_for_target(
                target=target,
                past_key_values=past_key_values_target,
                query_position_ids=block_position_ids,
                dtype=target.dtype,
            )
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            cache_position=block_position_ids[0],
            attention_mask=attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        if debug_expected_output_ids is not None and not debug_reported_verify_mismatch:
            for local_logit_idx in range(posterior.shape[1]):
                expected_abs_idx = start + local_logit_idx + 1
                if expected_abs_idx >= debug_expected_output_ids.shape[1]:
                    break
                prefix_end = start + local_logit_idx + 1
                if not torch.equal(
                    block_output_ids[0, : local_logit_idx + 1],
                    debug_expected_output_ids[0, start:prefix_end],
                ):
                    continue
                posterior_token = posterior[0, local_logit_idx]
                expected_token = debug_expected_output_ids[0, expected_abs_idx]
                if posterior_token != expected_token:
                    logits = output.logits[0, local_logit_idx].float()
                    top_values, top_indices = torch.topk(logits, k=min(5, logits.numel()))
                    top_summary = ",".join(
                        f"{int(token_id.item())}:{float(value.item()):.6f}"
                        for token_id, value in zip(top_indices, top_values)
                    )
                    expected_logit = float(logits[expected_token].item())
                    posterior_logit = float(logits[posterior_token].item())
                    draft_next = (
                        block_output_ids[0, local_logit_idx + 1]
                        if local_logit_idx + 1 < block_output_ids.shape[1]
                        else torch.tensor(-1, device=posterior.device, dtype=posterior.dtype)
                    )
                    label = "" if debug_label is None else f" {debug_label}"
                    print(
                        f"[DFLASH-VERIFY-MISMATCH]{label} "
                        f"round_start_abs={start} round_start_gen={start - num_input_tokens} "
                        f"logit_idx={local_logit_idx} predicts_abs={expected_abs_idx} "
                        f"predicts_gen={expected_abs_idx - num_input_tokens} "
                        f"expected={int(expected_token.item())} posterior={int(posterior_token.item())} "
                        f"draft_next={int(draft_next.item())} "
                        f"draft_next_matches_posterior={bool(draft_next == posterior_token)} "
                        f"expected_logit={expected_logit:.6f} posterior_logit={posterior_logit:.6f} "
                        f"top_logits={top_summary}",
                        flush=True,
                    )
                    debug_reported_verify_mismatch = True
                    break
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        _crop_shared_layers(past_key_values_target, start, past_key_values_target.get_seq_length())
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - acceptance_length - 1 : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    valid_length = min(max_length, start + 1)
    output_ids = output_ids[:, :valid_length]
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
