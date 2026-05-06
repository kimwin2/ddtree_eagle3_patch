import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import AutoConfig
from transformers.activations import ACT2FN


def _make_causal_mask(
    input_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    batch_size, target_length = input_shape
    mask = torch.full((target_length, target_length), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(target_length, past_key_values_length, dtype=dtype, device=device),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_length: Optional[int] = None) -> torch.Tensor:
    batch_size, source_length = mask.size()
    target_length = target_length if target_length is not None else source_length
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_length, source_length).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_length, head_dim = hidden_states.shape
    if repeats == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        repeats,
        seq_length,
        head_dim,
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * repeats, seq_length, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Eagle3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        positions = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class Eagle3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Eagle3Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        rope_theta = getattr(config, "rope_theta", 10000)
        self.rotary_emb = Eagle3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, query_length, _ = hidden_states.size()

        if getattr(self.config, "pretraining_tp", 1) > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp,
                dim=0,
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = torch.cat([F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            key_states = torch.cat([F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            value_states = torch.cat([F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_value_length = key_states.shape[-2]
        if past_key_value is not None:
            key_value_length += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=key_value_length)
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        present_key_value = (key_states, value_states) if use_cache else None

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, query_length, -1)

        if getattr(self.config, "pretraining_tp", 1) > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp))
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, present_key_value


class Eagle3MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "pretraining_tp", 1) > 1:
            slice_size = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_size, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_size, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_size, dim=1)
            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice_size, dim=2)
            return sum(F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = Eagle3Attention(config=config)
        self.mlp = Eagle3MLP(config)
        self.hidden_norm = Eagle3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = Eagle3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Eagle3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, self_attn_weights, present_key_value


class Eagle3DraftModel(nn.Module):
    def __init__(
        self,
        config,
        total_tokens: int = 32,
        depth: int = 7,
        top_k: int = 8,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        self.hidden_size = config.hidden_size
        self.embed_tokens: Optional[nn.Embedding] = None
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)
        self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)
        self.midlayer = Eagle3DecoderLayer(config)
        self.norm = Eagle3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        self.stable_kv = None
        self.tree_mask = None
        self.target_layer_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)

    @property
    def device(self) -> torch.device:
        return self.fc.weight.device

    def tie_target_embeddings(self, embed_tokens: nn.Embedding) -> None:
        self.embed_tokens = embed_tokens
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self) -> None:
        if self.embed_tokens is None:
            raise RuntimeError("Eagle3DraftModel must be tied to target embeddings before generation.")
        device = self.embed_tokens.weight.device
        self.tree_mask_init = torch.eye(self.top_k, device=device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=device, dtype=torch.long)

    def reset(self) -> None:
        self.tree_mask = None

    def reset_kv(self) -> None:
        self.stable_kv = None

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> Optional[torch.Tensor]:
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                torch.Size(input_shape),
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask,
                torch.float32,
                target_length=input_shape[-1],
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        if self.tree_mask is not None:
            _, _, tree_shape0, tree_shape1 = self.tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][self.tree_mask == 0] = torch.finfo(torch.float32).min
        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]], None]:
        if self.embed_tokens is None:
            raise RuntimeError("Eagle3DraftModel must be tied to target embeddings before generation.")

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if inputs_embeds is None:
            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past += past_key_values_length
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=hidden_states.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        dtype = self.fc.weight.dtype
        inputs_embeds = inputs_embeds.to(dtype)
        hidden_states = hidden_states.to(dtype)
        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        next_decoder_cache = () if use_cache else None
        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True,
        )
        if use_cache:
            next_decoder_cache += (layer_outputs[2],)
        return layer_outputs[0], next_decoder_cache, None

    @torch.no_grad()
    def topk_generate(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        inputs_embeds: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        scores_list = []
        parents_list = []
        speculative_tokens = []

        input_ids = input_ids.to(hidden_states.device)
        sample_token = input_ids[:, -1]
        input_ids = input_ids[:, 1:]
        self.initial_position_id = input_ids.shape[1]
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:, 1:]
            assert input_ids.shape[1] == inputs_embeds.shape[1]

        self.reset()
        last_hidden, past_key_values = self._get_initial_hidden(hidden_states, input_ids, inputs_embeds)
        self.stable_kv = past_key_values

        topk_index, scores = self._get_topk_tokens(last_hidden)
        scores_list.append(scores[None] if len(scores.shape) == 1 else scores)
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))

        if self.config.vocab_size == self.draft_vocab_size:
            speculative_tokens.append(topk_index)
            input_ids = topk_index
        else:
            mapped_tokens = topk_index + self.d2t[topk_index]
            speculative_tokens.append(mapped_tokens)
            input_ids = mapped_tokens

        input_hidden = last_hidden[None].repeat(1, self.top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(self.top_k, device=self.device)

        for level in range(self.depth):
            tree_mask, input_hidden, input_ids, scores, topk_cs_index, past_key_values = self._process_tree_level(
                level,
                tree_mask,
                input_hidden,
                input_ids,
                scores,
                topk_cs_index,
                scores_list,
                parents_list,
                speculative_tokens,
                past_key_values,
            )

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self._finalize_results(
            scores_list,
            speculative_tokens,
            sample_token,
            parents_list,
        )
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    def _get_initial_hidden(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        inputs_embeds: Optional[Tensor] = None,
    ) -> tuple[Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            outputs = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                inputs_embeds=(inputs_embeds[:, kv_len:] if inputs_embeds is not None else None),
                past_key_values=self.stable_kv,
                use_cache=True,
            )
        else:
            outputs = self(
                hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
        out_hidden, past_key_values, _ = outputs
        return out_hidden[:, -1], past_key_values

    def _get_topk_tokens(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.lm_head(self.norm(hidden))
        probs = self.logsoftmax(logits)
        topk = torch.topk(probs, self.top_k, dim=-1)
        return topk.indices, topk.values[0]

    def _process_tree_level(
        self,
        level: int,
        tree_mask: Tensor,
        input_hidden: Tensor,
        input_ids: Tensor,
        scores: Tensor,
        topk_cs_index: Tensor,
        scores_list: list[Tensor],
        parents_list: list[Tensor],
        speculative_tokens: list[Tensor],
        past_key_values,
    ):
        self.tree_mask = tree_mask
        position_ids = self.position_ids + self.initial_position_id
        out_hidden, past_key_values, _ = self(
            input_hidden,
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )
        self.initial_position_id += 1

        bias1 = self.top_k if level > 0 else 0
        bias2 = max(0, level - 1)
        bias = 1 + self.top_k**2 * bias2 + bias1
        parents = topk_cs_index + bias
        parents_list.append(parents)

        topk_index, topk_p = self._get_topk_tokens(out_hidden[0])
        cu_scores = topk_p + (scores[:, None] if len(scores.shape) == 1 else scores)
        topk_cs = torch.topk(cu_scores.view(-1), self.top_k, dim=-1)
        topk_cs_index, scores = topk_cs.indices, topk_cs.values

        out_ids = topk_cs_index // self.top_k
        input_hidden = out_hidden[:, out_ids]
        input_ids = topk_index.view(-1)[topk_cs_index][None]

        if self.config.vocab_size == self.draft_vocab_size:
            speculative_tokens.append(topk_index)
        else:
            input_ids = input_ids + self.d2t[input_ids]
            speculative_tokens.append(topk_index + self.d2t[topk_index])

        scores_list.append(cu_scores)
        tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
        return tree_mask, input_hidden, input_ids, scores, topk_cs_index, past_key_values

    def _finalize_results(
        self,
        scores_list: list[Tensor],
        speculative_tokens: list[Tensor],
        sample_token: Tensor,
        parents_list: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        all_scores = torch.cat(scores_list, dim=0).view(-1)
        all_tokens = torch.cat(speculative_tokens, dim=0).view(-1)
        top_scores = torch.topk(all_scores, self.total_tokens, dim=-1)
        top_indices = torch.sort(top_scores.indices).values
        draft_tokens = all_tokens[top_indices]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        tree_mask, tree_position_ids = self._build_tree_mask(top_indices, parents_list)
        retrieve_indices = self._generate_retrieve_indices(tree_position_ids, top_indices, parents_list)
        return draft_tokens[None], retrieve_indices, tree_mask, tree_position_ids

    def _build_tree_mask(self, top_indices: Tensor, parents_list: list[Tensor]) -> tuple[Tensor, Tensor]:
        all_parents = torch.cat(parents_list, dim=0)[top_indices // self.top_k].long()
        mask_index = torch.searchsorted(top_indices, all_parents - 1, right=False)
        mask_index[all_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask = torch.eye(self.total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for index in range(self.total_tokens):
            tree_mask[index + 1] |= tree_mask[mask_index_list[index]]
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        return tree_mask.float()[None, None], tree_position_ids

    def _generate_retrieve_indices(
        self,
        tree_position_ids: Tensor,
        top_indices: Tensor,
        parents_list: list[Tensor],
    ) -> Tensor:
        all_parents = torch.cat(parents_list, dim=0)[top_indices // self.top_k].long()
        mask_index = torch.searchsorted(top_indices, all_parents - 1, right=False)
        mask_index[all_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        noleaf_index = torch.unique(mask_index).tolist()
        leaf_num = self.total_tokens - (len(noleaf_index) - 1)
        retrieve_indices = torch.zeros(leaf_num, torch.max(tree_position_ids).item() + 1, dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()
        position_ids_list = tree_position_ids.tolist()

        row_index = 0
        for index in range(self.total_tokens + 1):
            if index not in noleaf_index:
                current_index = index
                depth = position_ids_list[index]
                for path_index in reversed(range(depth + 1)):
                    retrieve_indices[row_index][path_index] = current_index
                    current_index = mask_index_list[current_index - 1]
                row_index += 1
        return torch.tensor(retrieve_indices, dtype=torch.long)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        total_tokens: int = 32,
        depth: int = 7,
        top_k: int = 8,
        threshold: float = 1.0,
        dtype: Optional[torch.dtype] = None,
    ) -> "Eagle3DraftModel":
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config=config, total_tokens=total_tokens, depth=depth, top_k=top_k, threshold=threshold)
        state_dict = _load_eagle3_state_dict(pretrained_model_name_or_path)
        model.load_state_dict(state_dict, strict=False)
        if dtype is not None:
            model.to(dtype=dtype)
        return model


def _load_eagle3_state_dict(pretrained_model_name_or_path: str) -> dict[str, torch.Tensor]:
    candidates = ("model.safetensors", "pytorch_model.bin")
    for filename in candidates:
        if os.path.isdir(pretrained_model_name_or_path):
            weight_path = Path(pretrained_model_name_or_path) / filename
            if not weight_path.exists():
                continue
            weight_path = str(weight_path)
        else:
            try:
                weight_path = hf_hub_download(pretrained_model_name_or_path, filename)
            except Exception:
                continue
        if filename.endswith(".safetensors"):
            return load_file(weight_path, device="cpu")
        return torch.load(weight_path, map_location="cpu")
    raise FileNotFoundError(f"Eagle3 weights not found in {pretrained_model_name_or_path}")
