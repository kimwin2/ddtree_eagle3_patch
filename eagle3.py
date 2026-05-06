from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from ddtree import compact_dynamic_cache
from dflash import cuda_time, empty_stage_times
from model import Eagle3DraftModel, sample


EAGLE3_STAGE_ORDER = ("draft", "verify", "commit")


def eagle3_target_layer_ids(target_config, draft_model: Eagle3DraftModel) -> list[int]:
    if draft_model.target_layer_ids is not None:
        return list(draft_model.target_layer_ids)
    num_hidden_layers = getattr(target_config, "num_hidden_layers")
    return [2, num_hidden_layers // 2, num_hidden_layers - 3]


def extract_eagle3_context_feature(
    hidden_states: tuple[torch.Tensor, ...],
    target_layer_ids: list[int],
) -> torch.Tensor:
    return torch.cat([hidden_states[layer_id] for layer_id in target_layer_ids], dim=-1)


def compile_eagle3_attention_mask(
    tree_mask: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tree_mask = tree_mask.to(device=device, dtype=torch.bool)
    current_length = tree_mask.shape[-1]
    attention_mask = torch.zeros(
        (1, 1, current_length, past_length + current_length),
        dtype=dtype,
        device=device,
    )
    tree_block = attention_mask[:, :, :, past_length:]
    tree_block.masked_fill_(~tree_mask, torch.finfo(dtype).min)
    return attention_mask


def evaluate_eagle3_posterior(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    temperature: float,
) -> tuple[int, int, torch.Tensor, torch.Tensor, list[int]]:
    device = logits.device
    retrieve_indices = retrieve_indices.to(device=device)
    padded_draft_tokens = torch.cat(
        [
            draft_tokens.to(device=device),
            torch.full((1, 1), -1, dtype=draft_tokens.dtype, device=device),
        ],
        dim=1,
    )
    candidates = padded_draft_tokens[0, retrieve_indices]

    node_posterior = sample(logits, temperature)[0]
    gather_indices = retrieve_indices.clamp(min=0)
    posterior_paths = node_posterior[gather_indices[:, :-1]]
    valid_candidate_tokens = retrieve_indices[:, 1:] >= 0
    posterior_mask = (candidates[:, 1:] == posterior_paths) & valid_candidate_tokens
    candidates_accept_length = posterior_mask.int().cumprod(dim=1).sum(dim=1)
    accept_length = int(candidates_accept_length.max().item())
    if accept_length == 0:
        best_candidate = 0
    else:
        best_candidate = int(torch.argmax(candidates_accept_length).item())

    accepted_indices_tensor = retrieve_indices[best_candidate, : accept_length + 1]
    accepted_indices = [int(index) for index in accepted_indices_tensor.tolist()]
    last_accepted_index = accepted_indices[-1]
    sample_token = sample(logits[:, last_accepted_index : last_accepted_index + 1, :], temperature)
    return best_candidate, accept_length, sample_token, candidates, accepted_indices


@torch.inference_mode()
def target_generate(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    position_ids = torch.arange(max_length + 1, device=target.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=target.device)

    past_key_values = DynamicCache()
    stage_times = empty_stage_times(("decode",))

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )
    next_token = sample(output.logits, temperature).to(target.device)
    output_ids = torch.cat([input_ids, next_token], dim=1)
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    round_timestamps = []
    first_token_is_stop = (
        stop_token_ids_tensor is not None
        and torch.isin(next_token[0], stop_token_ids_tensor).any()
    )
    while output_ids.shape[1] < max_length and not first_token_is_stop:
        decode_stage_start = cuda_time()
        current_position = output_ids.shape[1] - 1
        output = target(
            next_token,
            position_ids=position_ids[:, current_position : current_position + 1],
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = sample(output.logits, temperature).to(target.device)
        output_ids = torch.cat([output_ids, next_token], dim=1)
        stage_times["decode"] += cuda_time() - decode_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if stop_token_ids_tensor is not None and torch.isin(next_token[0], stop_token_ids_tensor).any():
            break

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
        acceptance_lengths=[1] * num_output_tokens,
        decode_rounds=num_output_tokens,
        stage_times=stage_times,
        round_timestamps=round_timestamps,
    )


@torch.inference_mode()
def eagle3_generate(
    model: Eagle3DraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    position_ids = torch.arange(max_length + model.total_tokens + 8, device=target.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=target.device)
    target_layer_ids = eagle3_target_layer_ids(target.config, model)

    past_key_values_target = DynamicCache()
    stage_times = empty_stage_times(EAGLE3_STAGE_ORDER)
    model.reset_kv()

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    next_root_token = sample(output.logits, temperature).to(target.device)
    target_hidden = extract_eagle3_context_feature(output.hidden_states, target_layer_ids).to(model.device)
    time_to_first_token = cuda_time() - prefill_start

    draft_stage_start = cuda_time()
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.topk_generate(
        target_hidden,
        torch.cat((input_ids, next_root_token), dim=1).to(model.device),
    )
    stage_times["draft"] += cuda_time() - draft_stage_start

    committed_ids = input_ids.clone()
    decode_start = cuda_time()
    round_clock_start = cuda_time()
    acceptance_lengths = []
    round_timestamps = []

    while committed_ids.shape[1] < max_length:
        past_length = committed_ids.shape[1]
        draft_tokens = draft_tokens.to(target.device)
        verify_position_ids = tree_position_ids.to(target.device).unsqueeze(0) + past_length
        verify_attention_mask = compile_eagle3_attention_mask(
            tree_mask,
            past_length=past_length,
            dtype=target.dtype,
            device=target.device,
        )

        verify_stage_start = cuda_time()
        output = target(
            draft_tokens,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        best_candidate, accept_length, sample_token, candidates, accepted_indices = evaluate_eagle3_posterior(
            output.logits,
            draft_tokens,
            retrieve_indices,
            temperature,
        )

        remaining_tokens = max_length - committed_ids.shape[1]
        if len(accepted_indices) > remaining_tokens:
            accepted_indices = accepted_indices[:remaining_tokens]
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=target.device)
        accepted_tokens = candidates[None, best_candidate, : len(accepted_indices)].to(target.device)
        committed_ids = torch.cat([committed_ids, accepted_tokens], dim=1)
        compact_dynamic_cache(past_key_values_target, past_length, accepted_indices)

        hidden_state_new = extract_eagle3_context_feature(output.hidden_states, target_layer_ids).to(model.device)
        accepted_hidden = hidden_state_new.index_select(1, accepted_index_tensor.to(model.device))
        acceptance_lengths.append(len(accepted_indices))
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)

        if (
            len(accepted_indices) == remaining_tokens
            or (
                stop_token_ids_tensor is not None
                and torch.isin(accepted_tokens[0], stop_token_ids_tensor).any()
            )
        ):
            break

        draft_stage_start = cuda_time()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.topk_generate(
            accepted_hidden,
            torch.cat((committed_ids, sample_token.to(target.device)), dim=1).to(model.device),
        )
        stage_times["draft"] += cuda_time() - draft_stage_start

    output_ids = committed_ids[:, :max_length]
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
