import heapq
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from loguru import logger
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import (
    DFlashDraftModel,
    apply_logit_processing,
    compute_target_lm_logits,
    embed_target_input_ids,
    get_model_text_config,
    sample,
    extract_context_feature,
)
from dflash import dflash_generate, cuda_time, empty_stage_times, format_top_logits


DDTREE_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")
DDTREE_TREE_BUILD_STAGE_ORDER = ("tree_build_copy", "tree_build_heap", "tree_build_visibility")


_CPP_COMPACT_ENABLED = False


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Ignoring invalid integer value for {name}={value!r}")
        return default


def _ddtree_debug_path() -> Path:
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", rank)
    template = os.environ.get("DDTREE_DEBUG_JSONL", "ddtree_debug_rank{rank}.jsonl")
    path_text = template.format(rank=rank, local_rank=local_rank, pid=os.getpid())
    path = Path(path_text)
    if "{rank}" not in template and "{local_rank}" not in template and "{pid}" not in template:
        world_size = _env_int("WORLD_SIZE", 1)
        if world_size > 1:
            path = path.with_name(f"{path.stem}.rank{rank}{path.suffix or '.jsonl'}")
    return path


def _write_ddtree_debug(record: dict) -> None:
    path = _ddtree_debug_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _tensor_int_list(tensor: torch.Tensor, limit: int | None = None) -> list[int]:
    flat = tensor.detach().reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    return [int(value) for value in flat.cpu().tolist()]


def _topk_debug_rows(logits: torch.Tensor, rows: int, topk: int) -> list[dict]:
    if logits.numel() == 0 or rows <= 0 or topk <= 0:
        return []
    rows = min(rows, logits.shape[0])
    topk = min(topk, logits.shape[-1])
    values, indices = torch.topk(logits[:rows].float(), k=topk, dim=-1)
    values_cpu = values.detach().cpu().tolist()
    indices_cpu = indices.detach().cpu().tolist()
    return [
        {
            "row": row_idx,
            "ids": [int(token_id) for token_id in indices_cpu[row_idx]],
            "logits": [float(value) for value in values_cpu[row_idx]],
        }
        for row_idx in range(rows)
    ]


def _tensor_stats(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None:
        return None
    data = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(data.mean().item()) if data.numel() else 0.0,
        "std": float(data.std(unbiased=False).item()) if data.numel() else 0.0,
        "norm": float(data.norm().item()) if data.numel() else 0.0,
        "absmax": float(data.abs().max().item()) if data.numel() else 0.0,
    }


def _tensor_probe(tensor: torch.Tensor | None, limit: int = 8) -> dict | None:
    stats = _tensor_stats(tensor)
    if stats is None:
        return None
    data = tensor.detach().float().reshape(-1)
    if data.numel():
        stats.update({
            "sum": float(data.sum().item()),
            "head": [float(value) for value in data[:limit].cpu().tolist()],
            "tail": [float(value) for value in data[-limit:].cpu().tolist()],
        })
    else:
        stats.update({"sum": 0.0, "head": [], "tail": []})
    return stats


def _cache_seq_length(cache: DynamicCache | None) -> int | None:
    if cache is None or not hasattr(cache, "get_seq_length"):
        return None
    return int(cache.get_seq_length())


def _selected_topk_debug_rows(
    logits: torch.Tensor,
    rows: list[int] | tuple[int, ...],
    topk: int,
) -> list[dict]:
    if logits.numel() == 0 or topk <= 0:
        return []
    row_ids = sorted({int(row) for row in rows if 0 <= int(row) < logits.shape[0]})
    if not row_ids:
        return []
    row_tensor = torch.tensor(row_ids, dtype=torch.long, device=logits.device)
    values, indices = torch.topk(
        logits.index_select(0, row_tensor).float(),
        k=min(topk, logits.shape[-1]),
        dim=-1,
    )
    values_cpu = values.detach().cpu().tolist()
    indices_cpu = indices.detach().cpu().tolist()
    return [
        {
            "row": row_id,
            "ids": [int(token_id) for token_id in indices_cpu[offset]],
            "logits": [float(value) for value in values_cpu[offset]],
        }
        for offset, row_id in enumerate(row_ids)
    ]


def _mask_counts_summary(mask: torch.Tensor, current_length: int, past_length: int) -> dict:
    kv_length = past_length + current_length
    visible = mask[0, 0, :current_length, :kv_length] == 0
    tree_visible = visible[:, past_length:kv_length]
    prefix_visible = visible[:, :past_length]
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "kv_length": int(kv_length),
        "full_visible_counts": _tensor_int_list(visible.sum(dim=-1)),
        "prefix_visible_counts": _tensor_int_list(prefix_visible.sum(dim=-1)),
        "tree_visible_counts": _tensor_int_list(tree_visible.sum(dim=-1)),
    }


def _compact_attention_debug(
    attention_mask: torch.Tensor,
    target_attention_mask: torch.Tensor | dict[str, torch.Tensor],
    verify_position_ids: torch.Tensor,
    past_length: int,
    current_length: int,
) -> dict:
    kv_length = past_length + current_length
    key_positions = torch.arange(
        kv_length,
        dtype=verify_position_ids.dtype,
        device=verify_position_ids.device,
    )
    key_positions[past_length:kv_length] = verify_position_ids[0, :current_length]
    summary = {
        "base": _mask_counts_summary(attention_mask, current_length, past_length),
        "query_positions": _tensor_int_list(verify_position_ids[0, :current_length]),
        "key_positions_tree": _tensor_int_list(key_positions[past_length:kv_length]),
        "key_positions_prefix_tail": _tensor_int_list(
            key_positions[max(0, past_length - 8) : past_length]
        ),
    }
    if isinstance(target_attention_mask, dict):
        summary["target_keys"] = sorted(target_attention_mask.keys())
        for key, value in target_attention_mask.items():
            summary[key] = _mask_counts_summary(value, current_length, past_length)
    else:
        summary["target_keys"] = ["tensor"]
        summary["target"] = _mask_counts_summary(
            target_attention_mask,
            current_length,
            past_length,
        )
    return summary


def _cache_debug_summary(cache: DynamicCache | None, max_layers: int = 8) -> dict | None:
    if cache is None:
        return None

    layers = []
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        for layer_idx, key in enumerate(key_cache[:max_layers]):
            value = value_cache[layer_idx]
            if key is None or value is None or key.numel() == 0:
                layers.append({"layer": layer_idx, "empty": True})
                continue
            layers.append({
                "layer": layer_idx,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "seq_len": int(key.shape[-2]),
                "key_dtype": str(key.dtype),
            })
        return {
            "seq_length": int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else None,
            "num_layers": len(key_cache),
            "layers": layers,
        }

    cache_layers = getattr(cache, "layers", None)
    if cache_layers is not None:
        for layer_idx, layer in enumerate(cache_layers[:max_layers]):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None or key.numel() == 0:
                layers.append({"layer": layer_idx, "empty": True})
                continue
            layers.append({
                "layer": layer_idx,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "seq_len": int(key.shape[-2]),
                "key_dtype": str(key.dtype),
            })
        return {
            "seq_length": int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else None,
            "num_layers": len(cache_layers),
            "layers": layers,
        }

    return {"unsupported": type(cache).__name__}


def _mask_visible_summary(mask: torch.Tensor, current_length: int, past_length: int) -> dict:
    kv_length = past_length + current_length
    mask_block = mask[0, 0, :current_length, :kv_length]
    visible = mask_block == 0
    tree_visible = visible[:, past_length:kv_length]
    prefix_visible = visible[:, :past_length]
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "kv_length": int(kv_length),
        "full_visible_counts": _tensor_int_list(visible.sum(dim=-1)),
        "prefix_visible_counts": _tensor_int_list(prefix_visible.sum(dim=-1)),
        "tree_visible_counts": _tensor_int_list(tree_visible.sum(dim=-1)),
        "tree_visible_indices": [
            _tensor_int_list(torch.nonzero(tree_visible[row_idx], as_tuple=False).flatten())
            for row_idx in range(current_length)
        ],
    }


def _attention_mask_debug_summary(
    attention_mask: torch.Tensor,
    target_attention_mask: torch.Tensor | dict[str, torch.Tensor],
    verify_position_ids: torch.Tensor,
    past_length: int,
    current_length: int,
) -> dict:
    kv_length = past_length + current_length
    key_positions = torch.arange(kv_length, dtype=verify_position_ids.dtype, device=verify_position_ids.device)
    key_positions[past_length:kv_length] = verify_position_ids[0, :current_length]

    summary = {
        "base": _mask_visible_summary(attention_mask, current_length, past_length),
        "query_positions": _tensor_int_list(verify_position_ids[0, :current_length]),
        "key_positions_prefix_tail": _tensor_int_list(key_positions[max(0, past_length - 8) : past_length]),
        "key_positions_tree": _tensor_int_list(key_positions[past_length:kv_length]),
    }
    if isinstance(target_attention_mask, dict):
        summary["target_keys"] = sorted(target_attention_mask.keys())
        for key, value in target_attention_mask.items():
            summary[key] = _mask_visible_summary(value, current_length, past_length)
    else:
        summary["target_keys"] = ["tensor"]
        summary["target"] = _mask_visible_summary(target_attention_mask, current_length, past_length)
    return summary


@lru_cache(maxsize=1)
def load_cpp_compact_module():
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:
        logger.warning(f"torch.utils.cpp_extension is unavailable; falling back to Python cache compaction. {exc}")
        return None

    cpp_source = r"""
torch::Tensor compact_tail_inplace(torch::Tensor cache_tensor, int64_t past_length, torch::Tensor keep_current_indices) {
    TORCH_CHECK(cache_tensor.dim() >= 2, "cache_tensor must have rank >= 2");
    TORCH_CHECK(keep_current_indices.dim() == 1, "keep_current_indices must be a 1D tensor");
    TORCH_CHECK(keep_current_indices.scalar_type() == torch::kLong, "keep_current_indices must have dtype torch.long");
    TORCH_CHECK(cache_tensor.device() == keep_current_indices.device(), "cache_tensor and keep_current_indices must be on the same device");

    const int64_t seq_dim = cache_tensor.dim() - 2;
    TORCH_CHECK(past_length >= 0, "past_length must be non-negative");
    TORCH_CHECK(past_length <= cache_tensor.size(seq_dim), "past_length exceeds cache sequence length");

    const int64_t current_length = cache_tensor.size(seq_dim) - past_length;
    if (current_length <= 0) {
        return cache_tensor;
    }

    const int64_t keep_count = keep_current_indices.numel();
    TORCH_CHECK(keep_count >= 0, "keep_count must be non-negative");
    TORCH_CHECK(keep_count <= current_length, "keep_count exceeds appended window length");

    if (keep_count == 0 || keep_count == current_length) {
        return cache_tensor;
    }

    auto tail = cache_tensor.narrow(seq_dim, past_length, current_length);
    auto kept_tail = tail.index_select(seq_dim, keep_current_indices);
    cache_tensor.narrow(seq_dim, past_length, keep_count).copy_(kept_tail);
    return cache_tensor;
}
"""
    try:
        module = load_inline(
            name="ddtree_compact_tail_ext_v1",
            cpp_sources=[cpp_source],
            functions=["compact_tail_inplace"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("Loaded inline C++ tail cache compaction extension for DDTree.")
        return module
    except Exception as exc:
        logger.warning(
            f"Failed to build inline C++ tail cache compaction extension; falling back to Python implementation. {exc}"
        )
        return None


def maybe_enable_cpp_compact(enabled: bool) -> None:
    global _CPP_COMPACT_ENABLED
    _CPP_COMPACT_ENABLED = enabled
    if enabled:
        load_cpp_compact_module()


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float]]:
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    heap_start = time.perf_counter()
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[depth - 1, rank]) + float(top_log_probs_np[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes


def compile_ddtree_tree(
    root_token_id: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_length > 0:
        attention_mask_buffer[0, 0, :previous_tree_length, previous_tree_start : previous_tree_start + previous_tree_length] = 0

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[0, 0, :current_length, past_length : past_length + current_length]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    attention_mask = attention_mask_buffer[:, :, :current_length, : past_length + current_length]
    return verify_input_ids, verify_position_ids, attention_mask, past_length, current_length


def prepare_ddtree_attention_mask_for_target(
    target: AutoModelForCausalLM,
    attention_mask: torch.Tensor,
    verify_position_ids: torch.Tensor,
    past_length: int,
    current_length: int,
) -> torch.Tensor | dict[str, torch.Tensor]:
    target_config = get_model_text_config(target)
    layer_types = getattr(target_config, "layer_types", None)
    if not layer_types or "sliding_attention" not in layer_types:
        return attention_mask

    sliding_window = getattr(target_config, "sliding_window", None)
    if sliding_window is None or sliding_window <= 0:
        return {
            "full_attention": attention_mask,
            "sliding_attention": attention_mask,
        }

    kv_length = past_length + current_length
    key_positions = torch.arange(kv_length, dtype=verify_position_ids.dtype, device=verify_position_ids.device)
    key_positions[past_length:kv_length] = verify_position_ids[0, :current_length]
    query_positions = verify_position_ids[0, :current_length]
    sliding_visible = (
        (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
        & (key_positions.unsqueeze(0) > (query_positions.unsqueeze(1) - int(sliding_window)))
    )

    sliding_attention_mask = attention_mask.clone()
    sliding_attention_mask[0, 0, :current_length, :kv_length].masked_fill_(
        ~sliding_visible,
        torch.finfo(attention_mask.dtype).min,
    )
    return {
        "full_attention": attention_mask,
        "sliding_attention": sliding_attention_mask,
    }


def should_rebuild_ddtree_target_cache(target: AutoModelForCausalLM) -> bool:
    target_config = get_model_text_config(target)
    model_type = str(getattr(target_config, "model_type", ""))
    layer_types = getattr(target_config, "layer_types", None)
    return model_type.startswith("gemma4") or bool(layer_types and "sliding_attention" in layer_types)


def rebuild_ddtree_target_cache(
    target: AutoModelForCausalLM,
    output_ids: torch.Tensor,
    position_ids: torch.Tensor,
    end: int,
) -> DynamicCache:
    rebuilt_cache = DynamicCache()
    target(
        output_ids[:, :end],
        position_ids=position_ids[:, :end],
        past_key_values=rebuilt_cache,
        use_cache=True,
        logits_to_keep=1,
    )
    return rebuilt_cache


def follow_verified_tree(child_maps: list[dict[int, int]], posterior: torch.Tensor) -> tuple[list[int], int]:
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def _compact_appended_window(
    cache_tensor: torch.Tensor,
    past_length: int,
    keep_current_indices: torch.Tensor,
    expected_current_length: int | None = None,
) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if expected_current_length is not None and current_length != expected_current_length:
        raise RuntimeError(
            "DDTree cache compaction expects a full dense target cache whose "
            "physical KV tail is exactly the verified tree window. Got "
            f"past_length={past_length}, cache_seq_len={cache_tensor.shape[-2]}, "
            f"tail_length={current_length}, expected_tail_length={expected_current_length}. "
            "For Gemma4, create the target cache with DynamicCache() rather than "
            "DynamicCache(config=...), because HF sliding-window cache layers only "
            "store a window tail and cannot be compacted by absolute DDTree slots."
        )
    if current_length <= 0:
        return

    keep_count = keep_current_indices.numel()
    if keep_count > current_length:
        raise RuntimeError(
            f"DDTree accepted {keep_count} cache indices from a verified tail of "
            f"length {current_length}."
        )
    if keep_count > 0:
        max_keep_index = int(keep_current_indices.max().item())
        min_keep_index = int(keep_current_indices.min().item())
        if min_keep_index < 0 or max_keep_index >= current_length:
            raise RuntimeError(
                "DDTree accepted cache index outside the verified tail: "
                f"min={min_keep_index}, max={max_keep_index}, tail_length={current_length}."
            )
    if keep_count == 0:
        return
    if keep_count == current_length:
        identity_indices = torch.arange(
            current_length,
            dtype=keep_current_indices.dtype,
            device=keep_current_indices.device,
        )
        if bool(torch.equal(keep_current_indices, identity_indices)):
            return
        kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(-2, keep_current_indices)
        cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)
        return

    if _CPP_COMPACT_ENABLED:
        module = load_cpp_compact_module()
        if module is not None:
            module.compact_tail_inplace(cache_tensor, past_length, keep_current_indices)
            return

    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(-2, keep_current_indices)
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(
    past_key_values: DynamicCache,
    past_length: int,
    keep_current_indices: list[int],
    expected_current_length: int | None = None,
) -> None:
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(keep_current_indices, dtype=torch.long, device=device)
        return keep_tensor_by_device[device]

    seen_cache_views: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()

    def cache_view_key(cache_tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
        return (
            cache_tensor.untyped_storage().data_ptr(),
            cache_tensor.storage_offset(),
            tuple(cache_tensor.shape),
            tuple(cache_tensor.stride()),
        )

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            if key_cache is None or value_cache is None or key_cache.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(key_cache.device)
            for cache_tensor in (key_cache, value_cache):
                view_key = cache_view_key(cache_tensor)
                if view_key in seen_cache_views:
                    continue
                seen_cache_views.add(view_key)
                _compact_appended_window(
                    cache_tensor,
                    past_length,
                    keep_tensor,
                    expected_current_length=expected_current_length,
                )
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            for cache_tensor in (layer.keys, layer.values):
                view_key = cache_view_key(cache_tensor)
                if view_key in seen_cache_views:
                    continue
                seen_cache_views.add(view_key)
                _compact_appended_window(
                    cache_tensor,
                    past_length,
                    keep_tensor,
                    expected_current_length=expected_current_length,
                )
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for DDTree cache compaction.")


@torch.inference_mode()
def ddtree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    save_tree_traces: bool = False,
    debug_expected_output_ids: torch.Tensor | None = None,
    debug_label: str = "",
    debug_mismatch_log_limit: int | None = 8,
    on_commit=None,
) -> SimpleNamespace:
    if block_size <= 1:
        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            debug_expected_output_ids=debug_expected_output_ids,
            debug_label=debug_label,
            debug_mismatch_log_limit=debug_mismatch_log_limit,
            on_commit=on_commit,
        )

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = draft_horizon if tree_budget is None else max(tree_budget, 0)
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DDTREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0
    debug_mismatch_count = 0
    debug_mismatch_suppressed = False
    ddtree_compare_debug_enabled = _env_flag("DDTREE_DEBUG_COMPARE")
    if ddtree_compare_debug_enabled and not _env_flag("DDTREE_DEBUG_COMPARE_MISMATCH"):
        debug_mismatch_log_limit = 0

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
                f"[DDTREE-VERIFY-MISMATCH-SUPPRESSED] {debug_label} "
                f"further mismatch logs suppressed after {debug_mismatch_log_limit} messages",
                flush=True,
            )
            debug_mismatch_suppressed = True

    ddtree_debug_enabled = ddtree_compare_debug_enabled
    ddtree_debug_include_unlabeled = _env_flag("DDTREE_DEBUG_COMPARE_INCLUDE_UNLABELED")
    ddtree_debug_max_rounds = _env_int("DDTREE_DEBUG_COMPARE_MAX_ROUNDS", 3)
    ddtree_debug_topk = _env_int("DDTREE_DEBUG_COMPARE_TOPK", 8)
    ddtree_debug_depths = _env_int("DDTREE_DEBUG_COMPARE_DEPTHS", draft_horizon)

    def should_emit_ddtree_debug(round_idx: int) -> bool:
        if not ddtree_debug_enabled:
            return False
        if not debug_label and not ddtree_debug_include_unlabeled:
            return False
        return ddtree_debug_max_rounds < 0 or round_idx < ddtree_debug_max_rounds

    while start < max_length:
        round_idx = len(acceptance_lengths)
        emit_round_debug = should_emit_ddtree_debug(round_idx)
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        noise_embedding = embed_target_input_ids(target, block_output_ids)
        draft_cache_seq_before = past_key_values_draft.get_seq_length()
        draft_position_ids = position_ids[:, draft_cache_seq_before : start + block_size]
        draft_context_hidden = target_hidden
        draft_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_position_ids,
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        draft_sample_hidden = draft_hidden[:, -draft_horizon:, :]
        draft_logits = compute_target_lm_logits(target, draft_sample_hidden)
        draft_logits = apply_logit_processing(draft_logits, model.logit_scale, model.final_logit_softcapping)
        past_key_values_draft.crop(start)
        draft_cache_seq_after_crop = past_key_values_draft.get_seq_length()
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_ddtree_tree(
            draft_logits[0], tree_budget
        )
        stage_times["tree_build"] += cuda_time() - tree_build_start
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = cuda_time()
        last_tree_start = previous_tree_start
        last_tree_length = previous_tree_length
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        # Verify appends the whole tree to the live cache. Commit then compacts
        # the appended window down to the accepted root-to-leaf path.
        verify_cache = past_key_values_target
        target_cache_seq_before_verify = _cache_seq_length(verify_cache)
        target_attention_mask = prepare_ddtree_attention_mask_for_target(
            target=target,
            attention_mask=verify_attention_mask,
            verify_position_ids=verify_position_ids,
            past_length=previous_tree_start,
            current_length=previous_tree_length,
        )
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=target_attention_mask,
            past_key_values=verify_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start
        target_cache_seq_after_verify = _cache_seq_length(verify_cache)

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(
            accepted_indices,
            dtype=torch.long,
            device=verify_input_ids.device,
        )
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)
        verify_target_hidden = extract_context_feature(
            output.hidden_states,
            model.target_layer_ids,
        )
        target_hidden = verify_target_hidden.index_select(1, accepted_index_tensor)

        compact_dynamic_cache(
            verify_cache,
            start,
            accepted_indices,
            expected_current_length=previous_tree_length,
        )
        past_key_values_target = verify_cache
        target_cache_seq_after_compact = _cache_seq_length(verify_cache)

        if emit_round_debug:
            expected_window = None
            if debug_expected_output_ids is not None:
                expected_ids = debug_expected_output_ids.to(device=output.logits.device)
                expected_end = min(expected_ids.shape[1], start + previous_tree_length + 1)
                expected_window = _tensor_int_list(expected_ids[0, start:expected_end])

            target_probe_rows = sorted({
                0,
                max(0, previous_tree_length - 1),
                *[int(index) for index in accepted_indices],
                *[max(0, int(index) - 1) for index in accepted_indices],
            })
            target_config = get_model_text_config(target)
            target_dtype = str(next(target.parameters()).dtype)
            draft_dtype = str(next(model.parameters()).dtype)
            _write_ddtree_debug({
                "event": "ddtree_compare_round",
                "schema": 1,
                "label": debug_label,
                "rank": int(os.environ.get("RANK", "0")),
                "round": round_idx,
                "num_input_tokens": int(num_input_tokens),
                "start": int(start),
                "generated_start": int(start - num_input_tokens),
                "block_size": int(block_size),
                "draft_horizon": int(draft_horizon),
                "tree_budget": int(tree_budget),
                "temperature": float(temperature),
                "logit_scale": float(model.logit_scale),
                "final_logit_softcapping": (
                    None
                    if model.final_logit_softcapping is None
                    else float(model.final_logit_softcapping)
                ),
                "target_layer_ids": [int(layer_id) for layer_id in model.target_layer_ids],
                "target_layer_types": list(getattr(target_config, "layer_types", []) or []),
                "target_sliding_window": getattr(target_config, "sliding_window", None),
                "target_attn_implementation": getattr(target_config, "_attn_implementation", None),
                "draft_attn_implementation": getattr(model.config, "_attn_implementation", None),
                "target_param_dtype": target_dtype,
                "draft_param_dtype": draft_dtype,
                "target_hidden_dtype": str(draft_context_hidden.dtype),
                "noise_embedding_dtype": str(noise_embedding.dtype),
                "draft_hidden_dtype": str(draft_sample_hidden.dtype),
                "draft_logits_dtype": str(draft_logits.dtype),
                "target_logits_dtype": str(output.logits.dtype),
                "last_tree_start": int(last_tree_start),
                "last_tree_length": int(last_tree_length),
                "current_tree_start": int(previous_tree_start),
                "current_tree_length": int(previous_tree_length),
                "draft_cache_seq_before": int(draft_cache_seq_before),
                "draft_cache_seq_after_crop": int(draft_cache_seq_after_crop),
                "target_cache_seq_before_verify": target_cache_seq_before_verify,
                "target_cache_seq_after_verify": target_cache_seq_after_verify,
                "target_cache_seq_after_compact": target_cache_seq_after_compact,
                "draft_position_ids": _tensor_int_list(draft_position_ids[0]),
                "root_token": int(root_token[0, 0].item()),
                "input_ids_tail": _tensor_int_list(output_ids[0, max(0, start - 32) : start + 1]),
                "block_input_ids": _tensor_int_list(block_output_ids[0, :block_size]),
                "verify_input_ids": _tensor_int_list(verify_input_ids[0, :previous_tree_length]),
                "verify_position_ids": _tensor_int_list(verify_position_ids[0, :previous_tree_length]),
                "node_token_ids": [int(token_id) for token_id in node_token_ids.tolist()],
                "node_depths": [int(depth) for depth in node_depths.tolist()],
                "parents": [int(parent) for parent in parents],
                "child_maps": [
                    {str(int(token)): int(child) for token, child in child_map.items()}
                    for child_map in child_maps
                ],
                "target_hidden_before_draft": _tensor_probe(draft_context_hidden),
                "noise_embedding": _tensor_probe(noise_embedding),
                "draft_sample_hidden": _tensor_probe(draft_sample_hidden),
                "draft_logits": _tensor_probe(draft_logits),
                "draft_topk": _topk_debug_rows(
                    draft_logits[0],
                    rows=min(ddtree_debug_depths, draft_logits.shape[1]),
                    topk=ddtree_debug_topk,
                ),
                "mask": _compact_attention_debug(
                    verify_attention_mask,
                    target_attention_mask,
                    verify_position_ids,
                    previous_tree_start,
                    previous_tree_length,
                ),
                "posterior": _tensor_int_list(posterior[0, :previous_tree_length]),
                "target_logits": _tensor_probe(output.logits[:, :previous_tree_length]),
                "target_topk_rows": _selected_topk_debug_rows(
                    output.logits[0, :previous_tree_length],
                    rows=target_probe_rows,
                    topk=ddtree_debug_topk,
                ),
                "accepted_indices": [int(index) for index in accepted_indices],
                "accepted_tokens": _tensor_int_list(accepted_tokens[0]),
                "accepted_length": int(len(accepted_indices)),
                "next_token": int(next_token),
                "expected_window": expected_window,
                "verify_target_hidden": _tensor_stats(verify_target_hidden),
                "accepted_target_hidden": _tensor_probe(target_hidden),
            })

        if debug_expected_output_ids is not None:
            expected_ids = debug_expected_output_ids.to(device=output.logits.device)
            for local_idx in range(1, len(accepted_indices)):
                commits_abs = start + local_idx
                if commits_abs >= expected_ids.shape[1]:
                    break
                expected = expected_ids[0, commits_abs]
                actual = accepted_tokens[0, local_idx]
                if bool((actual != expected).item()):
                    logit_idx = accepted_indices[local_idx - 1]
                    node_index = accepted_indices[local_idx]
                    path_tokens = accepted_tokens[0].tolist()
                    emit_verify_mismatch(
                        f"[DDTREE-VERIFY-MISMATCH] {debug_label} "
                        f"round_start_abs={start} round_start_gen={start - num_input_tokens} "
                        f"node_index={node_index} depth={local_idx} "
                        f"commits_abs={commits_abs} commits_gen={commits_abs - num_input_tokens} "
                        f"expected={int(expected.item())} committed={int(actual.item())} "
                        f"accepted_indices={[int(index) for index in accepted_indices]} "
                        f"next_token={int(next_token)} "
                        f"expected_logit={float(output.logits[0, logit_idx, expected].float().item()):.6f} "
                        f"committed_logit={float(output.logits[0, logit_idx, actual].float().item()):.6f} "
                        f"top_logits={format_top_logits(output.logits[0, logit_idx])} "
                        f"path_tokens={[int(token) for token in path_tokens]}"
                    )
                    break
            else:
                predicts_abs = start + len(accepted_indices)
                if predicts_abs < expected_ids.shape[1]:
                    expected = expected_ids[0, predicts_abs]
                    actual = torch.tensor(next_token, dtype=expected.dtype, device=expected.device)
                    if bool((actual != expected).item()):
                        logit_idx = accepted_indices[-1]
                        final_node_index = accepted_indices[-1]
                        path_tokens = accepted_tokens[0].tolist()
                        emit_verify_mismatch(
                            f"[DDTREE-VERIFY-MISMATCH] {debug_label} "
                            f"round_start_abs={start} round_start_gen={start - num_input_tokens} "
                            f"node_index={final_node_index} depth={len(accepted_indices) - 1} "
                            f"predicts_abs={predicts_abs} predicts_gen={predicts_abs - num_input_tokens} "
                            f"expected={int(expected.item())} posterior={int(actual.item())} "
                            f"accepted_indices={[int(index) for index in accepted_indices]} "
                            f"expected_logit={float(output.logits[0, logit_idx, expected].float().item()):.6f} "
                            f"posterior_logit={float(output.logits[0, logit_idx, actual].float().item()):.6f} "
                            f"top_logits={format_top_logits(output.logits[0, logit_idx])} "
                            f"path_tokens={[int(token) for token in path_tokens]}"
                        )

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        if on_commit is not None:
            committed = output_ids[0, start : start + len(accepted_indices)].tolist()
            on_commit([int(token_id) for token_id in committed], len(accepted_indices))

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "accepted_indices": [int(index) for index in accepted_indices],
                "tree": {
                    "node_token_ids": [int(token_id) for token_id in node_token_ids.tolist()],
                    "node_depths": [int(depth) for depth in node_depths.tolist()],
                    "parents": [int(parent) for parent in parents],
                },
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
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
        round_trees=round_trees,
    )
