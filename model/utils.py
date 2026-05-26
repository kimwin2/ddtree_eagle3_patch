import math
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset, Features, Sequence, Value, concatenate_datasets
import json
import re
_AGENT_TARGET_FEATURES = Features({"turns": Sequence(Value("large_string"))})

_TAU2_TASK_URLS = {
    "airline": "https://huggingface.co/datasets/HuggingFaceH4/tau2-bench-data/resolve/main/domains/airline/tasks.json",
    "retail": "https://huggingface.co/datasets/HuggingFaceH4/tau2-bench-data/resolve/main/domains/retail/tasks.json",
}

_BFCL_V3_MULTITURN_URL = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
    "resolve/main/BFCL_v3_multi_turn_base.json"
)

def _to_pretty_text(value, max_chars: int = 6000) -> str:
    """Robustly stringify nested HF examples without exploding prompt length."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            text = str(value)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated for speed benchmark prompt]"
    return text


def _flatten_tau2_user_scenario(example: dict) -> str:
    scenario = example.get("user_scenario", "")
    if not isinstance(scenario, dict):
        return _to_pretty_text(scenario)

    instructions = scenario.get("instructions", scenario)
    if isinstance(instructions, dict):
        parts = []
        for key in ("domain", "reason_for_call", "known_info", "unknown_info", "task_instructions"):
            val = instructions.get(key)
            if val not in (None, ""):
                parts.append(f"{key}: {_to_pretty_text(val, max_chars=2000)}")
        if parts:
            return "\n".join(parts)

    return _to_pretty_text(scenario)


def _format_tau2_prompt(example: dict, domain: str) -> str:
    task_id = example.get("id", "")
    description = _to_pretty_text(example.get("description", ""), max_chars=2500)
    user_scenario = _flatten_tau2_user_scenario(example)
    initial_state = _to_pretty_text(example.get("initial_state", ""), max_chars=5000)
    eval_criteria = _to_pretty_text(example.get("evaluation_criteria", ""), max_chars=2500)

    return (
        f"You are a function-calling customer-support agent for the TAU-bench/{domain} domain.\n"
        "This is a speed-only benchmark prompt. Do not solve with natural language only; "
        "use tool-call style reasoning and emit tool calls when needed.\n\n"
        "Return format when calling a tool:\n"
        '{"tool_name": "<name>", "arguments": {"<arg>": "<value>"}}\n\n'
        f"Task id: {task_id}\n\n"
        f"Task description:\n{description}\n\n"
        f"User scenario:\n{user_scenario}\n\n"
        f"Initial state / database slice:\n{initial_state}\n\n"
        f"Success criteria:\n{eval_criteria}\n\n"
        "Start by deciding which tool/API calls are needed, then produce the next assistant action."
    )


def _load_tau2_domain(domain: str):
    if domain not in _TAU2_TASK_URLS:
        raise ValueError(f"Unsupported tau2 domain: {domain}. Supported: {sorted(_TAU2_TASK_URLS)}")

    raw = load_dataset(
        "json",
        data_files={"test": _TAU2_TASK_URLS[domain]},
        split="test",
    )

    return raw.map(
        lambda x: {"turns": [_format_tau2_prompt(x, domain)]},
        remove_columns=raw.column_names,
        features=_AGENT_TARGET_FEATURES,
    )


def _flatten_bfcl_question(question) -> str:
    if isinstance(question, str):
        return question

    if not isinstance(question, list):
        return _to_pretty_text(question)

    turns = []
    for idx, turn in enumerate(question, start=1):
        messages = turn if isinstance(turn, list) else [turn]
        msg_texts = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                msg_texts.append(f"{role}: {content}")
            else:
                msg_texts.append(_to_pretty_text(msg))

        turns.append(f"Turn {idx}:\n" + "\n".join(msg_texts))

    return "\n\n".join(turns)


def _format_bfcl_multiturn_prompt(example: dict) -> str:
    question = _flatten_bfcl_question(example.get("question", ""))
    involved_classes = _to_pretty_text(example.get("involved_classes", []), max_chars=1000)
    tool_path_hint = _to_pretty_text(example.get("path", []), max_chars=2000)
    initial_config = _to_pretty_text(example.get("initial_config", {}), max_chars=6000)

    return (
        "You are an agent that solves tasks by calling functions/tools.\n"
        "This is a speed-only BFCL-style multi-turn benchmark prompt. "
        "Use function-call style outputs whenever an action requires a tool.\n\n"
        "Return format when calling a tool:\n"
        '{"tool_name": "<Class.method>", "arguments": {"<arg>": "<value>"}}\n\n'
        f"Benchmark id: {example.get('id', '')}\n\n"
        f"Available tool classes:\n{involved_classes}\n\n"
        f"Tool/API path hint:\n{tool_path_hint}\n\n"
        f"Initial environment config:\n{initial_config}\n\n"
        f"Conversation:\n{question}\n\n"
        "Produce the next assistant action."
    )


def _load_bfcl_v3_multiturn():
    raw = load_dataset(
        "json",
        data_files={"test": _BFCL_V3_MULTITURN_URL},
        split="test",
    )

    return raw.map(
        lambda x: {"turns": [_format_bfcl_multiturn_prompt(x)]},
        remove_columns=raw.column_names,
        features=_AGENT_TARGET_FEATURES,
    )

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

def get_drafter_config(config) -> dict:
    drafter_config = {}
    eagle_config = getattr(config, "eagle_config", None)
    if isinstance(eagle_config, dict):
        drafter_config.update(eagle_config)
    dflash_config = getattr(config, "dflash_config", None)
    if isinstance(dflash_config, dict):
        drafter_config.update(dflash_config)
    return drafter_config

def get_dflash_target_layer_ids(
    config,
    target_num_layers: int,
    num_draft_layers: int,
) -> tuple[list[int], bool]:
    eagle_aux_layer_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
    if eagle_aux_layer_ids:
        return [int(layer_id) - 1 for layer_id in eagle_aux_layer_ids], True

    drafter_config = get_drafter_config(config)
    for key in ("target_layer_ids", "layer_ids"):
        if key in drafter_config:
            return [int(layer_id) for layer_id in drafter_config[key]], True

    return build_target_layer_ids(target_num_layers, num_draft_layers), False

def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
    early_exit: bool = False,
) -> torch.Tensor:
    offset = 1
    selected_states = []
    
    if early_exit:
        fix_id = layer_ids[2]
        for i, layer_id in enumerate(layer_ids):
            if i > 2:
                selected_states.append(hidden_states[fix_id + offset])
            else:
                selected_states.append(hidden_states[layer_id + offset])
    else:
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
    embeddings_module = get_input_embeddings_module(model)
    embeddings = embeddings_module(input_ids)
    scale = get_gemma4_embedding_scale(getattr(model, "config", None))
    if scale is None or hasattr(embeddings_module, "embed_scale") or hasattr(embeddings_module, "scalar_embed_scale"):
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

def get_logit_scale(config) -> float:
    dflash_config = getattr(config, "dflash_config", None) or {}
    value = dflash_config.get("logit_scale", getattr(config, "logit_scale", 1.0))
    return float(value)

def apply_logit_processing(
    logits: torch.Tensor,
    logit_scale: float = 1.0,
    final_logit_softcapping: Optional[float] = None,
) -> torch.Tensor:
    logits = apply_final_logit_softcapping(logits, final_logit_softcapping)
    if logit_scale != 1.0:
        logits = logits * logit_scale
    return logits

def apply_final_logit_softcapping(
    logits: torch.Tensor,
    final_logit_softcapping: Optional[float],
) -> torch.Tensor:
    if final_logit_softcapping is None:
        return logits
    return torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping

def load_and_process_dataset(data_name: str):
    if data_name in ("tau-bench-airline", "taubench-airline", "tau2-airline"):
        dataset = _load_tau2_domain("airline")

    elif data_name in ("tau-bench-retail", "taubench-retail", "tau2-retail"):
        dataset = _load_tau2_domain("retail")

    elif data_name in ("tau-bench", "taubench", "tau2-bench"):
        dataset = concatenate_datasets([
            _load_tau2_domain("airline"),
            _load_tau2_domain("retail"),
        ])

    elif data_name in ("bfcl-v3-multiturn", "bfcl-multiturn", "bfcl"):
        dataset = _load_bfcl_v3_multiturn()
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
