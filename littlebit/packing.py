from typing import Tuple

import torch


def binary_packer(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise TypeError("Input tensor must be int8.")

    n_rows, n_cols = tensor.shape
    words_per_row = (n_cols + 31) // 32
    pad = words_per_row * 32 - n_cols
    device = tensor.device

    if pad:
        pad_tensor = torch.ones((n_rows, pad), dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, pad_tensor], dim=1)

    bits = ((1 - tensor) // 2).int()
    bits = bits.reshape(tensor.size(0), tensor.size(1) // 32, 32)
    weights = (2 ** torch.arange(32, dtype=torch.int32, device=device)).int()
    packed = (bits * weights).sum(dim=2, dtype=torch.int32)
    return packed.contiguous()


def binary_unpacker(
    packed_tensor: torch.Tensor, original_shape: Tuple[int, int]
) -> torch.Tensor:
    if packed_tensor.dim() != 2:
        raise ValueError(
            f"Expected a rank-2 packed tensor, got shape {tuple(packed_tensor.shape)}."
        )

    n_rows, n_cols = original_shape
    words_per_row = (n_cols + 31) // 32
    expected_shape = (n_rows, words_per_row)
    if tuple(packed_tensor.shape) != expected_shape:
        raise ValueError(
            f"Packed tensor shape {tuple(packed_tensor.shape)} does not match expected {expected_shape}."
        )

    unpacked = torch.zeros(
        n_rows,
        words_per_row * 32,
        dtype=torch.int8,
        device=packed_tensor.device,
    )
    for word_idx in range(words_per_row):
        word_data = packed_tensor[:, word_idx]
        bits = (
            word_data.unsqueeze(1) >> torch.arange(32, device=packed_tensor.device)
        ) & 1
        unpacked[:, word_idx * 32 : (word_idx + 1) * 32] = bits.to(torch.int8)

    unpacked = unpacked[:, :n_cols]
    return (1 - 2 * unpacked).to(torch.int8)


def int2_packer(tensor: torch.Tensor, quant_min: int = -2) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise TypeError("Input tensor must be int8.")

    quant_max = quant_min + 3
    if tensor.numel() and (tensor.min() < quant_min or tensor.max() > quant_max):
        raise ValueError(
            f"Input tensor values must be in [{quant_min}, {quant_max}] for int2 packing."
        )

    n_rows, n_cols = tensor.shape
    words_per_row = (n_cols + 15) // 16
    pad = words_per_row * 16 - n_cols
    device = tensor.device

    if pad:
        pad_tensor = torch.zeros((n_rows, pad), dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, pad_tensor], dim=1)

    codes = (tensor - quant_min).to(torch.int32)
    codes = codes.reshape(n_rows, words_per_row, 16)
    shifts = (2 * torch.arange(16, dtype=torch.int32, device=device)).int()
    packed = (codes << shifts).sum(dim=2, dtype=torch.int32)
    return packed.contiguous()


def int2_unpacker(
    packed_tensor: torch.Tensor,
    original_shape: Tuple[int, int],
    quant_min: int = -2,
) -> torch.Tensor:
    if packed_tensor.dim() != 2:
        raise ValueError(
            f"Expected a rank-2 packed tensor, got shape {tuple(packed_tensor.shape)}."
        )

    n_rows, n_cols = original_shape
    words_per_row = (n_cols + 15) // 16
    expected_shape = (n_rows, words_per_row)
    if tuple(packed_tensor.shape) != expected_shape:
        raise ValueError(
            f"Packed tensor shape {tuple(packed_tensor.shape)} does not match expected {expected_shape}."
        )

    shifts = (
        2 * torch.arange(16, dtype=torch.int32, device=packed_tensor.device)
    ).int()
    unpacked = torch.empty(
        n_rows,
        words_per_row * 16,
        dtype=torch.int8,
        device=packed_tensor.device,
    )
    for word_idx in range(words_per_row):
        word_data = packed_tensor[:, word_idx].to(torch.int32)
        codes = (word_data.unsqueeze(1) >> shifts) & 0x3
        unpacked[:, word_idx * 16 : (word_idx + 1) * 16] = (
            codes.to(torch.int8) + quant_min
        )

    return unpacked[:, :n_cols].contiguous()
