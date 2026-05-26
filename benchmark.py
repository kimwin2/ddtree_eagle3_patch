import argparse
import random
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
import os

os.environ['HF_HOME'] = "./hf_cache"
os.environ['HF_DATASETS_CACHE'] = "./hf_cache/datasets"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, load_and_process_dataset, sample
from dflash import cuda_time, dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from littlebit import load_quantized_dflash_model



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


@torch.inference_mode()
def target_generate(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.empty((1, max_length), dtype=torch.long, device=target.device)
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = torch.arange(max_length, device=target.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=target.device)
    past_key_values = None

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    past_key_values = output.past_key_values
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = num_input_tokens + 1
    while start < max_length:
        output = target(
            output_ids[:, start - 1 : start],
            position_ids=position_ids[:, start - 1 : start],
            past_key_values=past_key_values,
            use_cache=True,
        )
        output_ids[:, start : start + 1] = sample(output.logits, temperature)
        past_key_values = output.past_key_values

        if stop_token_ids_tensor is not None and torch.isin(output_ids[0, start], stop_token_ids_tensor).any():
            start += 1
            break
        start += 1

    output_ids = output_ids[:, :start]
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
        stage_times={"decode": total_decode_time},
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="31,63")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--apply-ee", action="store_true")

    parser.add_argument("--draft-type", type=str, default="dflash", choices=["dflash", "littlebit_dflash"])
    parser.add_argument("--quant-mod", type=str, default="LittleBitLinear")
    parser.add_argument("--quant-func", type=str, default="STEBinary")
    parser.add_argument("--split-dim", type=int, default=1024)
    parser.add_argument("--eff-bit", type=float, default=0.5)
    parser.add_argument("--kv-factor", type=float, default=1.0)
    parser.add_argument("--min-split-dim", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--residual", action="store_true")
    
    args = parser.parse_args()

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
    draft_algorithm = "dflash"

    target_attn_implementation = "sdpa"
    draft_attn_implementation = "flex_attention"

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    method_key_to_tree_budget = {}
    block_size = args.block_size
    if args.draft_type == "littlebit_dflash":
        draft_model = load_quantized_dflash_model(
            args.draft_name_or_path,
            device=device,
            torch_dtype=torch.bfloat16,
            quant_args=args,
            attn_implementation=draft_attn_implementation,
        )
    else:
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=draft_attn_implementation,
            dtype=torch.bfloat16,
        ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    methods_to_run = ["dflash"]
    if not args.flash_attn:
        ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update({f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    stop_token_ids = get_stop_token_ids(tokenizer, target)
    logger.info(f"Using stop_token_ids={stop_token_ids}")
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))

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
        if method_key == "dflash":
            _ = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                apply_ee=args.apply_ee
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
                apply_ee=args.apply_ee
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
                if method_key == "dflash":
                    response[method_key] = dflash_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=stop_token_ids,
                        temperature=args.temperature,
                        apply_ee=args.apply_ee
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
                        apply_ee=args.apply_ee
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
