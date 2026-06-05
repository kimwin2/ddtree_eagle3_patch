"""Calibration for DFlash draft-model activation quantization.

Provides a ``CalibrationDataReader`` (sequence-level samples, with the sample
count and sequence length as separate knobs) and a driver that runs the
speculative-decoding draft path in *observe* mode so the activation observers
see both the prefill-derived and the decode-derived activation distributions,
then freezes the uint16 asymmetric per-tensor qparams.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
from transformers import DynamicCache

from .quant import format_quant_summary
from .utils import (
    apply_logit_processing,
    compute_target_lm_logits,
    embed_target_input_ids,
    extract_context_feature,
    load_and_process_dataset,
    sample,
)


def parse_eval_tasks(spec) -> list[tuple[str, Optional[int]]]:
    """Parse an eval-task spec into ``[(dataset_name, eval_max_samples), ...]``.

    Accepts either a comma-separated string ``"gsm8k:128,math500:128"`` (matching
    the ``run_benchmark.sh`` TASKS format) or an already-parsed list of tuples.
    A missing ``:max`` means the dataset is fully consumed by eval (no holdout).
    """
    if isinstance(spec, str):
        tasks: list[tuple[str, Optional[int]]] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            name, _, max_str = part.partition(":")
            name = name.strip()
            max_str = max_str.strip()
            tasks.append((name, int(max_str) if max_str else None))
        return tasks
    return [(name, (int(max_n) if max_n is not None else None)) for name, max_n in spec]


class CalibrationDataReader:
    """Yields tokenized calibration *sequences* (not single tokens).

    Each item is an ``input_ids`` tensor of shape ``(1, L)`` representing one
    prompt. The driver prefills the prompt and then decodes until the total
    length reaches the deployment ``seq_len``, so a single reader item exercises
    the full prefill + decode distribution.

    Two sources are supported (exactly one must be given):
        * ``dataset_name``: draw ``num_samples`` prompts from a single dataset.
        * ``eval_tasks``: pool *held-out* prompts (the samples NOT used by eval)
          across the evaluation datasets. This replicates eval's selection
          (``shuffle(seed).select(range(eval_max))``) and takes the remainder,
          so there is no calibration/eval data leakage. Datasets fully consumed
          by eval (``len <= eval_max``) contribute nothing.

    Args:
        tokenizer: HF tokenizer (used with the chat template).
        dataset_name: a name understood by ``load_and_process_dataset``.
        eval_tasks: eval-task spec (string or list); see ``parse_eval_tasks``.
        num_samples: number of sequences to emit (default 128).
        seq_len: deployment context length each sequence is grown to.
        prompt_len: tokens of prompt to keep per sample. Defaults to
            ``seq_len // 2`` so there is always room to exercise decode.
        seed: shuffling seed; must match eval's seed (0) for correct holdout.
        enable_thinking: passed to the chat template.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: Optional[str] = None,
        num_samples: int = 128,
        seq_len: int = 2048,
        prompt_len: Optional[int] = None,
        seed: int = 0,
        enable_thinking: bool = False,
        eval_tasks=None,
        verbose: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_samples = int(num_samples)
        self.seq_len = int(seq_len)
        self.prompt_len = int(prompt_len) if prompt_len is not None else max(1, self.seq_len // 2)
        self.enable_thinking = enable_thinking
        self.seed = int(seed)
        self.verbose = verbose
        self.source_datasets: list[str] = []
        self.dataset_counts: dict[str, int] = {}

        if eval_tasks is not None:
            self._prompts = self._build_heldout_prompts(parse_eval_tasks(eval_tasks))
        elif dataset_name is not None:
            dataset = load_and_process_dataset(dataset_name)
            if len(dataset) > self.num_samples:
                dataset = dataset.shuffle(seed=self.seed).select(range(self.num_samples))
            self._prompts = self._encode_dataset(dataset, limit=self.num_samples)
            self.source_datasets = [dataset_name]
            self.dataset_counts = {dataset_name: len(self._prompts)}
        else:
            raise ValueError("CalibrationDataReader requires either dataset_name or eval_tasks.")
        self._cursor = 0

    def _encode_one(self, instance) -> torch.Tensor:
        turns = instance["turns"]
        user_content = turns[0] if isinstance(turns, (list, tuple)) and turns else str(turns)
        input_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        # Keep the prompt within prompt_len so the decode stage still runs.
        if input_ids.shape[1] > self.prompt_len:
            input_ids = input_ids[:, : self.prompt_len]
        if input_ids.shape[1] >= self.seq_len:
            input_ids = input_ids[:, : max(1, self.seq_len - 1)]
        return input_ids

    def _encode_dataset(self, dataset, limit: Optional[int] = None) -> list[torch.Tensor]:
        prompts: list[torch.Tensor] = []
        for instance in dataset:
            prompts.append(self._encode_one(instance))
            if limit is not None and len(prompts) >= limit:
                break
        return prompts

    def _build_heldout_prompts(self, eval_tasks: list[tuple[str, Optional[int]]]) -> list[torch.Tensor]:
        named_pools: list[tuple[str, list[torch.Tensor]]] = []
        if self.verbose:
            print(
                f"[CALIB] building held-out calibration pool (seed={self.seed}, "
                f"target={self.num_samples} samples) from {len(eval_tasks)} eval datasets",
                flush=True,
            )
        for name, eval_max in eval_tasks:
            dataset = load_and_process_dataset(name)
            total = len(dataset)
            # Mirror eval: it only sub-selects when len(dataset) > eval_max.
            # Otherwise the whole dataset is used and there is no holdout.
            if eval_max is None or total <= eval_max:
                if self.verbose:
                    print(
                        f"[CALIB]   {name:<16} total={total:<6} eval_used={eval_max} "
                        f"holdout=0 (fully consumed by eval -> skipped)",
                        flush=True,
                    )
                continue
            held = dataset.shuffle(seed=self.seed).select(range(eval_max, total))
            held_total = len(held)
            # We never need more than num_samples from a single dataset; cap the
            # encoding work. The holdout is already shuffled, so the head of the
            # tail is a representative deterministic sample.
            if held_total > self.num_samples:
                held = held.select(range(self.num_samples))
            pool = self._encode_dataset(held)
            if self.verbose:
                print(
                    f"[CALIB]   {name:<16} total={total:<6} eval_used={eval_max:<6} "
                    f"holdout_available={held_total:<6} encoded={len(pool)}",
                    flush=True,
                )
            if pool:
                named_pools.append((name, pool))
                self.source_datasets.append(name)
        if not named_pools:
            raise ValueError(
                "No held-out calibration data: every eval dataset is fully consumed by eval. "
                "Reduce per-dataset eval max_samples or pass an explicit --calib-dataset."
            )
        # Round-robin interleave across datasets so the 128 samples stay diverse.
        prompts: list[torch.Tensor] = []
        counts: dict[str, int] = {name: 0 for name, _ in named_pools}
        idx = 0
        while len(prompts) < self.num_samples:
            progressed = False
            for name, pool in named_pools:
                if idx < len(pool):
                    prompts.append(pool[idx])
                    counts[name] += 1
                    progressed = True
                    if len(prompts) >= self.num_samples:
                        break
            if not progressed:
                break
            idx += 1
        self.dataset_counts = {name: count for name, count in counts.items() if count > 0}
        if self.verbose:
            composition = ", ".join(f"{name}={count}" for name, count in self.dataset_counts.items())
            print(
                f"[CALIB] held-out pool composition ({len(prompts)} samples): {composition}",
                flush=True,
            )
        return prompts

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._prompts)

    def get_next(self) -> Optional[torch.Tensor]:
        """ONNXRuntime-style sequential accessor; returns None when exhausted."""
        if self._cursor >= len(self._prompts):
            return None
        item = self._prompts[self._cursor]
        self._cursor += 1
        return item

    def rewind(self) -> None:
        self._cursor = 0


@torch.inference_mode()
def _observe_one_sequence(
    model,
    target,
    input_ids: torch.Tensor,
    seq_len: int,
    block_size: int,
    temperature: float,
) -> int:
    """Run the DFlash draft decode loop for one prompt, observing activations.

    Mirrors ``dflash.dflash_generate`` (block_size > 1 path) but runs to
    ``seq_len`` without early stopping so the observers see a full sequence.
    Returns the number of decode rounds executed.
    """
    device = target.device
    input_ids = input_ids.to(device)
    num_input_tokens = input_ids.shape[1]
    max_length = max(seq_len, num_input_tokens + block_size)
    mask_token_id = model.mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    start = num_input_tokens
    rounds = 0
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        noise_embedding = embed_target_input_ids(target, block_output_ids)
        draft_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        draft_logits = compute_target_lm_logits(target, draft_hidden[:, -block_size + 1 :, :])
        draft_logits = apply_logit_processing(draft_logits, model.logit_scale, model.final_logit_softcapping)
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits)

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[
            :, : acceptance_length + 1, :
        ]
        rounds += 1
    return rounds


@torch.inference_mode()
def calibrate_dflash_activations(
    model,
    target,
    reader: CalibrationDataReader,
    *,
    block_size: Optional[int] = None,
    temperature: float = 0.0,
    quant_config: Optional[dict] = None,
    enable_after: bool = True,
    progress: bool = True,
    print_summary: bool = True,
) -> dict:
    """Calibrate and freeze the draft model's activation quantizers.

    1. (Re)configures every FakeQuantize from ``quant_config``.
    2. Runs the draft decode loop over every calibration sequence in observe
       mode to collect activation statistics.
    3. Computes per-tensor asymmetric uint16 qparams and (optionally) enables
       fake quantization.

    Returns a small summary dict.
    """
    block_size = block_size if block_size is not None else model.block_size
    if block_size <= 1:
        raise ValueError("Activation calibration requires the draft path (block_size > 1).")

    num_configured = model.configure_activation_quant(quant_config)
    if print_summary:
        composition = ", ".join(f"{name}={count}" for name, count in reader.dataset_counts.items())
        print(
            f"[CALIB] starting observation: {len(reader)} sequences, seq_len={reader.seq_len}, "
            f"block_size={block_size}, {num_configured} quantizers | datasets: {composition}",
            flush=True,
        )

    was_training = model.training
    model.eval()
    model.set_activation_quant_enabled(False)
    model.set_activation_quant_observing(True)

    total_rounds = 0
    iterator = list(reader)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="calibrating dflash activations")
        except ImportError:
            pass
    for input_ids in iterator:
        total_rounds += _observe_one_sequence(
            model=model,
            target=target,
            input_ids=input_ids,
            seq_len=reader.seq_len,
            block_size=block_size,
            temperature=temperature,
        )

    model.set_activation_quant_observing(False)
    num_calibrated = model.finalize_activation_quant()
    if enable_after:
        model.set_activation_quant_enabled(True)
    if was_training:
        model.train()

    if print_summary:
        print(
            f"[CALIB] per-layer activation min/max after {len(reader)} sequences "
            f"({total_rounds} decode rounds):",
            flush=True,
        )
        print(format_quant_summary(model), flush=True)

    return {
        "num_quantizers": num_configured,
        "num_calibrated": num_calibrated,
        "num_sequences": len(reader),
        "decode_rounds": total_rounds,
        "seq_len": reader.seq_len,
        "prompt_len": reader.prompt_len,
        "datasets": list(reader.source_datasets),
        "dataset_counts": dict(reader.dataset_counts),
        "quant_config": quant_config or {},
    }
