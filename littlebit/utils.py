import argparse
import importlib
import inspect
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig

from model.dflash import DFlashDraftModel

from .modules import LittleBitLinear, LittleBitOnDeviceLinear
from .packing import binary_unpacker, int2_unpacker

__all__ = [
    "apply_littlebit_patch",
    "read_littlebit_config",
    "save_quantized_dflash_model",
    "load_quantized_dflash_model",
]


_LITTLEBIT_MODULES = {
    "LittleBitLinear": LittleBitLinear,
    "littlebit": LittleBitLinear,
    "default": LittleBitLinear,
    "LittleBitOnDeviceLinear": LittleBitOnDeviceLinear,
    "littlebit_on_device": LittleBitOnDeviceLinear,
    "on_device": LittleBitOnDeviceLinear,
}


def load_quant_fn(name: str):
    module = importlib.import_module("specforge.littlebit.functions")
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown LittleBit quant function: {name}") from exc


def load_quant_module(name: str | type[nn.Module]):
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name
    try:
        return _LITTLEBIT_MODULES[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown LittleBit quant module: {name}") from exc


def apply_littlebit_patch(
    model: nn.Module,
    quant_args,
    *,
    do_train: bool,
    exclude_names: list[str] | None = None,
):
    exclude_names = exclude_names or ["lm_head"]
    quant_func = load_quant_fn(getattr(quant_args, "quant_func", "STEBinary"))
    quant_mod = load_quant_module(
        getattr(quant_args, "quant_mod", "LittleBitLinear")
    )
    common_kwargs = {
        "do_train": do_train,
        "quant_func": quant_func,
        "residual": getattr(quant_args, "residual", False),
        "split_dim": getattr(quant_args, "split_dim", 1024),
        "eff_bit": getattr(quant_args, "eff_bit", 1.0),
        "min_split_dim": getattr(quant_args, "min_split_dim", 8),
        "group_size": getattr(quant_args, "group_size", 128),
    }
    kv_kwargs = {
        "ratio_factor": getattr(quant_args, "kv_factor", 1.0),
    }
    kv_patterns = [re.compile(r"\.k_proj$"), re.compile(r"\.v_proj$")]

    for name, module in model.named_modules():
        if any(excluded in name for excluded in exclude_names):
            continue
        if not isinstance(module, nn.Linear):
            continue

        current_kwargs = dict(common_kwargs)
        if any(pattern.search(name) for pattern in kv_patterns):
            current_kwargs.update(kv_kwargs)

        module.__class__ = quant_mod
        module.__quant_convert__(**current_kwargs)

    return model


def _load_raw_state_dict(model_path: str):
    state_dict = {}
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        shard_files = set(index["weight_map"].values())
        for shard_file in shard_files:
            with safe_open(
                os.path.join(model_path, shard_file), framework="pt", device="cpu"
            ) as handle:
                for key in handle.keys():
                    state_dict[key] = handle.get_tensor(key)
        return state_dict

    if os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state_dict[key] = handle.get_tensor(key)
        return state_dict

    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        kwargs = {"map_location": "cpu"}
        signature = inspect.signature(torch.load)
        if "mmap" in signature.parameters:
            kwargs["mmap"] = True
        if "weights_only" in signature.parameters:
            kwargs["weights_only"] = True
        try:
            return torch.load(bin_path, **kwargs)
        except Exception:
            kwargs["weights_only"] = False
            return torch.load(bin_path, **kwargs)

    raise FileNotFoundError(f"No model weights found in {model_path}")


def _load_and_process_state_dict(model_path: str, torch_dtype: torch.dtype):
    state_dict = _load_raw_state_dict(model_path)
    has_packed_weights = any(key.endswith("_packed") for key in state_dict.keys())
    if not has_packed_weights:
        return state_dict, False

    packed_components = defaultdict(dict)
    final_state_dict = {}
    pattern = re.compile(r"^(.*)\.([^.]+?)_(packed|shape|bit_width|quant_min)$")

    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            prefix, param_name, suffix_type = match.groups()
            packed_components[prefix][f"{param_name}_{suffix_type}"] = value
        else:
            final_state_dict[key] = value

    for prefix, components in packed_components.items():
        param_names = {
            key.rsplit("_", 1)[0] for key in components if key.endswith("_packed")
        }
        for name in param_names:
            packed_tensor = components.get(f"{name}_packed")
            shape_tensor = components.get(f"{name}_shape")
            if packed_tensor is None or shape_tensor is None:
                continue
            shape = tuple(shape_tensor.tolist())
            bit_width_tensor = components.get(f"{name}_bit_width")
            bit_width = (
                1 if bit_width_tensor is None else int(bit_width_tensor.item())
            )
            if bit_width == 1:
                unpacked = binary_unpacker(packed_tensor, shape).to(torch_dtype)
            elif bit_width == 2:
                quant_min_tensor = components.get(f"{name}_quant_min")
                quant_min = -2 if quant_min_tensor is None else int(
                    quant_min_tensor.item()
                )
                unpacked = int2_unpacker(
                    packed_tensor, shape, quant_min=quant_min
                ).to(torch_dtype)
            else:
                raise ValueError(f"Unsupported packed LittleBit bit width: {bit_width}")
            final_state_dict[f"{prefix}.{name}"] = unpacked

    return final_state_dict, True


def read_littlebit_config(model_path: str) -> dict:
    config_path = Path(model_path) / "littlebit_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _quant_config_dict(quant_args) -> dict:
    return {
        "quant_mod": getattr(quant_args, "quant_mod", "LittleBitLinear"),
        "quant_func": getattr(quant_args, "quant_func", "STEBinary"),
        "eff_bit": getattr(quant_args, "eff_bit", 1.0),
        "split_dim": getattr(quant_args, "split_dim", 1024),
        "residual": getattr(quant_args, "residual", False),
        "kv_factor": getattr(quant_args, "kv_factor", 1.0),
        "min_split_dim": getattr(quant_args, "min_split_dim", 8),
        "group_size": getattr(quant_args, "group_size", 128),
    }


def _load_state_dict_allow_meta(model: nn.Module, state_dict: dict, strict: bool = False):
    has_meta_tensor = any(param.is_meta for param in model.parameters()) or any(
        buffer.is_meta for buffer in model.buffers()
    )
    kwargs = {"strict": strict}
    if has_meta_tensor:
        signature = inspect.signature(model.load_state_dict)
        if "assign" not in signature.parameters:
            raise RuntimeError(
                "This checkpoint requires loading into meta parameters, but this "
                "PyTorch version does not support load_state_dict(assign=True)."
            )
        kwargs["assign"] = True

    missing, unexpected = model.load_state_dict(state_dict, **kwargs)
    remaining_meta = [
        name
        for name, param in model.named_parameters()
        if param is not None and param.is_meta
    ]
    remaining_meta.extend(
        name
        for name, buffer in model.named_buffers()
        if buffer is not None and buffer.is_meta
    )
    if remaining_meta:
        raise RuntimeError(
            "Some LittleBit tensors are still on meta after checkpoint load: "
            f"{remaining_meta[:10]}"
        )
    return missing, unexpected

def _get_draft_model_class(config):
    """Return the appropriate DFlash draft model class based on config."""
    model_type = getattr(config, "model_type", None)
    if model_type == "gauss3_dflash":
        return Gauss3DFlashDraftModel
    return DFlashDraftModel

def save_quantized_dflash_model(model: DFlashDraftModel, output_dir: str, quant_args):
    os.makedirs(output_dir, exist_ok=True)
    quant_config = _quant_config_dict(quant_args)
    for key, value in quant_config.items():
        setattr(model.config, key, value)
    model.config.quant_method = "littlebit"
    model.save_pretrained(output_dir)

    with open(
        os.path.join(output_dir, "littlebit_config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(quant_config, handle, indent=2)

    modeling_src = (
        Path(__file__).resolve().parent.parent / "modeling" / "draft" / "dflash.py"
    )
    modeling_dst = Path(output_dir) / "dflash.py"
    if modeling_src.exists():
        shutil.copy(modeling_src, modeling_dst)


def load_quantized_dflash_model(
    model_path: str,
    *,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    quant_args=None,
    do_train: bool = False,
    attn_implementation = "eager"
):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = attn_implementation
    if quant_args is None:
        config_dict = read_littlebit_config(model_path)
        quant_args = argparse.Namespace(**config_dict)

    model_cls = _get_draft_model_class(config)
    model = model_cls(config)
    
    model = apply_littlebit_patch(model, quant_args, do_train=do_train)
    state_dict, was_packed = _load_and_process_state_dict(model_path, torch_dtype)
    missing, unexpected = _load_state_dict_allow_meta(
        model, state_dict, strict=False
    )
    if missing:
        print(f"WARNING: Missing keys when loading quantized DFlash: {missing[:10]}")
    if unexpected:
        print(
            f"WARNING: Unexpected keys when loading quantized DFlash: {unexpected[:10]}"
        )

    if was_packed:
        for module in model.modules():
            if isinstance(module, LittleBitOnDeviceLinear):
                module.set_packed_mode(True, do_train=do_train)
            elif isinstance(module, LittleBitLinear):
                module._binarized = True

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch_dtype)
    model.train(do_train)
    return model
