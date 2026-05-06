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
from model import DFlashDraftModel, Eagle3DraftModel, load_and_process_dataset
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from eagle3 import eagle3_generate, target_generate


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
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--draft-algorithm", choices=["auto", "dflash", "eagle3"], default="auto")
    parser.add_argument("--eagle3-batch-size", type=int, default=1)
    parser.add_argument("--eagle3-depth", type=int, default=7)
    parser.add_argument("--eagle3-topk", type=int, default=8)
    parser.add_argument("--eagle3-tree-size", type=int, default=32)
    parser.add_argument("--save-path", type=str, default=None)
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
    if draft_algorithm == "dflash" and not installed_flash_attn:
        raise RuntimeError("flash_attn must be installed because the draft DFlash model always uses FlashAttention")

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = "flash_attention_2" if draft_algorithm == "dflash" else "pytorch"

    if draft_algorithm == "eagle3" and args.flash_attn:
        logger.warning("Eagle3 tree verification uses a custom attention mask; forcing the target verifier to torch.sdpa.")
        target_attn_implementation = "sdpa"
    elif not args.flash_attn and installed_flash_attn:
        logger.warning("DDTree uses a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa.")

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
    dataset = load_and_process_dataset(args.dataset)

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
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        if method_key == "eagle3":
            _ = eagle3_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                max_new_tokens=warmup_max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
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
                stop_token_ids=[tokenizer.eos_token_id],
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
                stop_token_ids=[tokenizer.eos_token_id],
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
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            for method_key in methods_to_run:
                if method_key == "eagle3":
                    response[method_key] = eagle3_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        stop_token_ids=[tokenizer.eos_token_id],
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
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
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
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
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
