import math
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset, Features, Sequence, Value

def get_text_config(config):
    return getattr(config, "text_config", config)

def is_gemma4_config(config) -> bool:
    text_config = get_text_config(config)
    return str(getattr(text_config, "model_type", "")).startswith("gemma4")

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids

def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden

def get_model_text_config(model):
    return get_text_config(getattr(model, "config", None))

def get_gemma4_embedding_scale(config) -> Optional[float]:
    text_config = get_text_config(config)
    if not is_gemma4_config(text_config):
        return None
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        return None
    return math.sqrt(float(hidden_size))

def get_input_embeddings_module(model):
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    for attr_path in (
        ("model", "embed_tokens"),
        ("language_model", "model", "embed_tokens"),
        ("language_model", "embed_tokens"),
    ):
        module = model
        for attr in attr_path:
            module = getattr(module, attr, None)
            if module is None:
                break
        if module is not None:
            return module
    raise AttributeError("Could not find input embeddings on the target model.")

def embed_target_input_ids(model, input_ids: torch.Tensor) -> torch.Tensor:
    embeddings = get_input_embeddings_module(model)(input_ids)
    scale = get_gemma4_embedding_scale(getattr(model, "config", None))
    if scale is None:
        return embeddings
    return embeddings * scale

def get_lm_head_module(model):
    for attr_path in (
        ("lm_head",),
        ("language_model", "lm_head"),
    ):
        module = model
        for attr in attr_path:
            module = getattr(module, attr, None)
            if module is None:
                break
        if module is not None:
            return module
    if hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            return lm_head
    raise AttributeError("Could not find lm_head/output embeddings on the target model.")

def compute_target_lm_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    lm_head = get_lm_head_module(model)
    if callable(lm_head):
        return lm_head(hidden_states)
    return F.linear(hidden_states, lm_head.weight, getattr(lm_head, "bias", None))

def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)

def get_final_logit_softcapping(config, target_config=None) -> Optional[float]:
    dflash_config = getattr(config, "dflash_config", None) or {}
    value = dflash_config.get("final_logit_softcapping", getattr(config, "final_logit_softcapping", None))
    if value is None and target_config is not None and is_gemma4_config(target_config):
        target_text_config = get_text_config(target_config)
        value = getattr(target_text_config, "final_logit_softcapping", None)
    if value is None:
        return None
    return float(value)

def apply_final_logit_softcapping(
    logits: torch.Tensor,
    final_logit_softcapping: Optional[float],
) -> torch.Tensor:
    if final_logit_softcapping is None:
        return logits
    return torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping

def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets 
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x['input'] else x['instruction'])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
    
    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]
        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"
        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features
        )
    
    return dataset
