import os
from math import ceil

import torch
import torch.nn as nn

from .packing import binary_packer, int2_packer


class LittleBitLinear(nn.Module):
    def __quant_convert__(
        self,
        do_train: bool,
        quant_func,
        *,
        split_dim: int = 1024,
        eff_bit: float | None = None,
        residual: bool = False,
        ratio_factor: float = 1.0,
        min_split_dim: int = 8,
        **kwargs,
    ):
        self.do_train = do_train
        self.quant_func = quant_func
        self.residual = residual
        self._binarized = False

        a, b = self.in_features, self.out_features
        split_calc_float = self._estimate_split_dim(a, b, eff_bit, residual)
        if split_calc_float is not None:
            split_calc_float *= ratio_factor
        final_split_dim = self._finalize_split_dim(
            split_calc_float, split_dim, min_split_dim
        )
        eff_bit_actual = self._compute_eff_bits(a, b, final_split_dim, residual)
        buffer_device = self._get_buffer_device()

        self.register_buffer(
            "_eff_bit_target",
            torch.tensor(
                -1.0 if eff_bit is None else float(eff_bit), device=buffer_device
            ),
        )
        self.register_buffer(
            "_split_dim_final",
            torch.tensor(final_split_dim, device=buffer_device),
        )
        self.register_buffer(
            "_eff_bit_actual",
            torch.tensor(eff_bit_actual, device=buffer_device),
        )
        self.split_dim = final_split_dim

        if self.do_train and hasattr(self, "weight") and self.weight is not None:
            self._initialize_parameters()
        else:
            self._initialize_empty_parameters()

    def _get_buffer_device(self):
        if hasattr(self, "weight") and self.weight is not None:
            return self.weight.device
        if self.bias is not None:
            return self.bias.device
        return torch.device("cpu")

    @staticmethod
    def _estimate_split_dim(a, b, eff_bit_target, residual) -> float | None:
        if eff_bit_target is None or a * b == 0:
            return None

        base = a + b + 16
        if residual:
            numerator = a * b * eff_bit_target - 32 * (a + b)
            denominator = 2 * base
        else:
            numerator = a * b * eff_bit_target - 16 * (a + b)
            denominator = base
        return numerator / denominator if denominator else None

    @staticmethod
    def _finalize_split_dim(
        split_float: float | None,
        split_default: int,
        min_split_dim: int,
    ) -> int:
        candidate = split_float if split_float is not None else split_default
        candidate = int(candidate) if candidate is not None else 0
        candidate = (candidate // 8) * 8
        if candidate == 0:
            candidate = min_split_dim
        return max(candidate, min_split_dim)

    @staticmethod
    def _compute_eff_bits(a: int, b: int, s: int, residual: bool) -> float:
        if a * b == 0:
            return float("inf")

        if residual:
            numerator = s * 2 * (a + b + 16) + 32 * (a + b)
        else:
            numerator = s * (a + b + 16) + 16 * (a + b)
        return numerator / (a * b)

    def forward(self, x):
        *seqlen, hidden_dim = x.shape
        output_shape = tuple(seqlen + [self.out_features])
        x = x.view(-1, hidden_dim)

        y = self._compute_forward(
            x, self.V, self.U, self.v2, self.v1, self.u2, self.u1
        )
        if self.residual:
            y = y + self._compute_forward(
                x, self.V_R, self.U_R, self.v2_R, self.v1_R, self.u2_R, self.u1_R
            )
        if self.bias is not None:
            y += self.bias
        return y.reshape(output_shape)

    def _compute_forward(self, x, V, U, v2, v1, u2, u1):
        Vq = self.quantize(V.to(x.dtype))
        Uq = self.quantize(U.to(x.dtype))
        v1u2 = v1 * u2
        return ((((x * v2) @ Vq.t()) * v1u2) @ Uq.t()) * u1

    def quantize(self, x):
        if self._binarized:
            return x
        return self.quant_func(x)

    def _initialize_empty_parameters(self):
        dtype = torch.bfloat16
        device = "meta"

        def create_param(*shape):
            return nn.Parameter(
                torch.empty(*shape, device=device, dtype=dtype),
                requires_grad=self.do_train,
            )

        self.U = create_param(self.out_features, self.split_dim)
        self.V = create_param(self.split_dim, self.in_features)
        self.u1 = create_param(1, self.out_features)
        self.u2 = create_param(1, self.split_dim)
        self.v1 = create_param(1, self.split_dim)
        self.v2 = create_param(1, self.in_features)

        if self.residual:
            self.U_R = create_param(self.out_features, self.split_dim)
            self.V_R = create_param(self.split_dim, self.in_features)
            self.u1_R = create_param(1, self.out_features)
            self.u2_R = create_param(1, self.split_dim)
            self.v1_R = create_param(1, self.split_dim)
            self.v2_R = create_param(1, self.in_features)

        if hasattr(self, "weight"):
            del self.weight
        self.register_parameter("weight", None)

    def _initialize_parameters(self):
        weight = (
            self.weight.data.float()
            if self.do_train and self.weight is not None
            else None
        )
        U, V, u1, u2, v1, v2 = self._decompose_matrix(weight)

        def create_param(tensor):
            return nn.Parameter(tensor, requires_grad=self.do_train)

        self.U = create_param(U)
        self.V = create_param(V)
        self.u1 = create_param(u1)
        self.u2 = create_param(u2)
        self.v1 = create_param(v1)
        self.v2 = create_param(v2)

        if self.residual:
            residual_weight = None
            if self.do_train:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                calc_device = (
                    torch.device(f"cuda:{local_rank}")
                    if torch.cuda.is_available()
                    else self.weight.device
                )

                U_g = self.quantize(U).to(calc_device)
                V_g = self.quantize(V).to(calc_device)
                u1_g, u2_g = u1.to(calc_device), u2.to(calc_device)
                v1_g, v2_g = v1.to(calc_device), v2.to(calc_device)
                approx = (U_g * (u1_g.t() @ u2_g)) @ (V_g * (v1_g.t() @ v2_g))
                residual_weight = self.weight.data.float() - approx.to(self.weight.device)
                del U_g, V_g, u1_g, u2_g, v1_g, v2_g, approx

            U_R, V_R, u1_R, u2_R, v1_R, v2_R = self._decompose_matrix(
                residual_weight
            )
            self.U_R = create_param(U_R)
            self.V_R = create_param(V_R)
            self.u1_R = create_param(u1_R)
            self.u2_R = create_param(u2_R)
            self.v1_R = create_param(v1_R)
            self.v2_R = create_param(v2_R)

        self.register_parameter("weight", None)
        self._binarized = False


    def _compute_itq_rotation(self, X, n_iter=20):
        """
        Finds optimal Rotation R that aligns data to the binary hypercube.
        Objective: min || B - X @ R ||_F^2  s.t. B = sign(X @ R)
        """
        with torch.no_grad():
            N, dim = X.shape
            device = X.device
            
            # 정밀도를 위해 잠시 float32 사용
            X_f = X.float()
            
            # 1. Initialize R with Random Orthogonal (기존 방식과 동일한 출발점)
            R = torch.empty((dim, dim), device=device, dtype=torch.float32)
            torch.nn.init.orthogonal_(R)
            
            # 2. Iterative Optimization (Alternating Minimization)
            for _ in range(n_iter):
                # Step A: R 고정, Binary Target B 업데이트
                Z = X_f @ R
                B = torch.sign(Z)
                
                # Step B: B 고정, R 업데이트 (Orthogonal Procrustes Problem)
                # Maximize Tr(B^T @ X @ R) -> SVD of B^T @ X
                M = B.t() @ X_f
                U_p, _, Vt_p = torch.linalg.svd(M, full_matrices=False)
                
                # Optimal R = V @ U^T
                R = Vt_p.t() @ U_p.t()
                
            return R.to(X.dtype)


    def _decompose_matrix(self, X=None, target_split_dim=None):
        if target_split_dim is None:
            target_split_dim = self.split_dim

        if self.do_train:
            X_f = X.float()
            U_t, S_t, Vh_t = torch.linalg.svd(X_f, full_matrices=False)

            S_sqrt_vec = torch.sqrt(S_t[:target_split_dim]) 
            S_sqrt_mat = torch.diag(S_sqrt_vec)

            U_temp = U_t[:, :target_split_dim] @ S_sqrt_mat  
            V_temp = S_sqrt_mat @ Vh_t[:target_split_dim, :]  

            with torch.no_grad():
                X_combined = torch.cat([U_temp, V_temp.t()], dim=0)
                R = self._compute_itq_rotation(X_combined, n_iter=50)
                U = (U_temp @ R).contiguous()
                V = (R.t() @ V_temp).contiguous()

            v1, v2 = self._rank_one_decompose(torch.abs(V))
            u1, u2 = self._rank_one_decompose(torch.abs(U))

            dtype = torch.bfloat16
            return (
                U.to(dtype), V.to(dtype), 
                u1.to(dtype), u2.to(dtype), 
                v1.to(dtype), v2.to(dtype)
            )
        else:
            U = torch.empty(self.out_features, target_split_dim)
            V = torch.empty(target_split_dim, self.in_features)
            u1 = torch.empty(1, self.out_features); u2 = torch.empty(1, target_split_dim)
            v1 = torch.empty(1, target_split_dim); v2 = torch.empty(1, self.in_features)
            return U, V, u1, u2, v1, v2

    def _rank_one_decompose(self, matrix, calc_device=None):
        original_device = matrix.device
        if calc_device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            calc_device = (
                torch.device(f"cuda:{local_rank}")
                if torch.cuda.is_available()
                else original_device
            )

        matrix_calc = matrix.to(calc_device)
        U, S, V = torch.svd_lowrank(matrix_calc, q=1)
        Vh = V.t()
        sqrt_s0 = torch.sqrt(S[0])
        u_component = (U[:, :1] * sqrt_s0).t().contiguous()
        v_component = (sqrt_s0 * Vh[:1, :]).contiguous()
        return u_component, v_component

    def pack_weights(self):
        packed = {}

        def pack_param(param, name):
            param_bin = self.quantize(param.data).to(torch.int8)
            packed[f"{name}_packed"] = binary_packer(param_bin)
            packed[f"{name}_shape"] = torch.tensor(param.shape, dtype=torch.long)

        pack_param(self.U, "U")
        pack_param(self.V, "V")
        if self.residual:
            pack_param(self.U_R, "U_R")
            pack_param(self.V_R, "V_R")
        return packed

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.get("prefix", "")
        state = super().state_dict(*args, **kwargs)
        keys_to_remove = [
            key
            for key in state.keys()
            if key.startswith(prefix + "U") or key.startswith(prefix + "V")
        ]
        for key in keys_to_remove:
            state.pop(key, None)

        packed_weights = self.pack_weights()
        for key, value in packed_weights.items():
            state[prefix + key] = value
        return state

    @property
    def eff_bit_target(self):
        value = self._eff_bit_target.item()
        return None if value < 0 else value

    @property
    def eff_bit_actual(self):
        return self._eff_bit_actual.item()

    @property
    def split_dim_used(self):
        return int(self._split_dim_final.item())

    @property
    def total_bit_usage(self):
        return self.eff_bit_actual * self.in_features * self.out_features


class LittleBitOnDeviceLinear(LittleBitLinear):
    INT2_QUANT_MIN = -2
    INT2_QUANT_MAX = 1
    RANK_MULTIPLE = 32

    def __quant_convert__(
        self,
        do_train: bool,
        quant_func=None,
        *,
        split_dim: int = 1024,
        eff_bit: float | None = None,
        residual: bool = False,
        ratio_factor: float = 1.0,
        min_split_dim: int = 32,
        group_size: int = 128,
        **kwargs,
    ):
        self.do_train = do_train
        self.quant_func = quant_func
        self.residual = residual
        self.group_size = int(group_size)
        self._binarized = False
        self._quantized = False

        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}.")

        a, b = self.in_features, self.out_features
        split_calc_float = self._estimate_split_dim(
            a, b, eff_bit, residual, self.group_size
        )
        if split_calc_float is not None:
            split_calc_float *= ratio_factor
        final_split_dim = self._finalize_split_dim(
            split_calc_float,
            split_dim,
            min_split_dim,
            max_rank=min(a, b),
        )
        if eff_bit is not None:
            final_split_dim = self._fit_split_dim_to_budget(
                a, b, final_split_dim, eff_bit, residual, self.group_size
            )
        eff_bit_actual = self._compute_eff_bits(
            a, b, final_split_dim, residual, self.group_size
        )
        buffer_device = self._get_buffer_device()

        self.register_buffer(
            "_eff_bit_target",
            torch.tensor(
                -1.0 if eff_bit is None else float(eff_bit), device=buffer_device
            ),
        )
        self.register_buffer(
            "_split_dim_final",
            torch.tensor(final_split_dim, device=buffer_device),
        )
        self.register_buffer(
            "_eff_bit_actual",
            torch.tensor(eff_bit_actual, device=buffer_device),
        )
        self.register_buffer(
            "_group_size",
            torch.tensor(self.group_size, device=buffer_device),
        )
        self.split_dim = final_split_dim

        if self.do_train and hasattr(self, "weight") and self.weight is not None:
            self._initialize_parameters()
        else:
            self._initialize_empty_parameters()

    @staticmethod
    def _estimate_split_dim(
        a, b, eff_bit_target, residual, group_size
    ) -> float | None:
        if eff_bit_target is None or a * b == 0:
            return None

        v_groups_per_rank = ceil(a / group_size)
        u_groups_per_rank = b / group_size
        bits_per_rank = 2 * (a + b) + 16 * (
            v_groups_per_rank + u_groups_per_rank
        )
        if residual:
            bits_per_rank *= 2
        return (a * b * eff_bit_target) / bits_per_rank if bits_per_rank else None

    @staticmethod
    def _finalize_split_dim(
        split_float: float | None,
        split_default: int,
        min_split_dim: int,
        *,
        max_rank: int,
    ) -> int:
        if max_rank <= 0:
            return 0

        min_rank = min(
            max(min_split_dim, LittleBitOnDeviceLinear.RANK_MULTIPLE), max_rank
        )
        candidate = split_float if split_float is not None else split_default
        candidate = int(candidate) if candidate is not None else 0
        candidate = (
            candidate // LittleBitOnDeviceLinear.RANK_MULTIPLE
        ) * LittleBitOnDeviceLinear.RANK_MULTIPLE
        if candidate == 0:
            candidate = min_rank
        candidate = min(candidate, max_rank)
        if max_rank >= LittleBitOnDeviceLinear.RANK_MULTIPLE:
            candidate = (
                candidate // LittleBitOnDeviceLinear.RANK_MULTIPLE
            ) * LittleBitOnDeviceLinear.RANK_MULTIPLE
        return max(candidate, min_rank)

    @staticmethod
    def _compute_eff_bits(
        a: int, b: int, s: int, residual: bool, group_size: int
    ) -> float:
        if a * b == 0:
            return float("inf")

        u_scale_count = b * ceil(s / group_size)
        v_scale_count = s * ceil(a / group_size)
        numerator = 2 * s * (a + b) + 16 * (u_scale_count + v_scale_count)
        if residual:
            numerator *= 2
        return numerator / (a * b)

    @classmethod
    def _fit_split_dim_to_budget(
        cls,
        a: int,
        b: int,
        split_dim: int,
        eff_bit_target: float,
        residual: bool,
        group_size: int,
    ) -> int:
        if split_dim <= cls.RANK_MULTIPLE:
            return split_dim

        min_rank = cls.RANK_MULTIPLE if min(a, b) >= cls.RANK_MULTIPLE else min(a, b)
        while split_dim > min_rank:
            eff_bits = cls._compute_eff_bits(a, b, split_dim, residual, group_size)
            if eff_bits <= eff_bit_target:
                break
            split_dim -= cls.RANK_MULTIPLE
        return split_dim

    def forward(self, x):
        *seqlen, hidden_dim = x.shape
        output_shape = tuple(seqlen + [self.out_features])
        x = x.view(-1, hidden_dim)

        y = self._compute_forward(x, self.V, self.U, self.V_scale, self.U_scale)
        if self.residual:
            y = y + self._compute_forward(
                x, self.V_R, self.U_R, self.V_R_scale, self.U_R_scale
            )
        if self.bias is not None:
            y += self.bias
        return y.reshape(output_shape)

    def _compute_forward(self, x, V, U, V_scale, U_scale):
        Vq = self._group_quantize(V.to(x.dtype), V_scale.to(x.dtype))
        Uq = self._group_quantize(U.to(x.dtype), U_scale.to(x.dtype))
        return (x @ Vq.t()) @ Uq.t()

    def _initialize_empty_parameters(self):
        dtype = torch.bfloat16
        device = "meta"

        def create_param(*shape):
            return nn.Parameter(
                torch.empty(*shape, device=device, dtype=dtype),
                requires_grad=self.do_train,
            )

        self.U = create_param(self.out_features, self.split_dim)
        self.V = create_param(self.split_dim, self.in_features)
        self.U_scale = create_param(
            self.out_features, self._num_groups(self.split_dim)
        )
        self.V_scale = create_param(
            self.split_dim, self._num_groups(self.in_features)
        )

        if self.residual:
            self.U_R = create_param(self.out_features, self.split_dim)
            self.V_R = create_param(self.split_dim, self.in_features)
            self.U_R_scale = create_param(
                self.out_features, self._num_groups(self.split_dim)
            )
            self.V_R_scale = create_param(
                self.split_dim, self._num_groups(self.in_features)
            )

        if hasattr(self, "weight"):
            del self.weight
        self.register_parameter("weight", None)

    def _initialize_parameters(self):
        weight = (
            self.weight.data.float()
            if self.do_train and self.weight is not None
            else None
        )
        U, V = self._decompose_matrix(weight)

        def create_param(tensor):
            return nn.Parameter(tensor, requires_grad=self.do_train)

        self.U = create_param(U)
        self.V = create_param(V)
        self.U_scale = create_param(self._init_group_scale(U))
        self.V_scale = create_param(self._init_group_scale(V))

        if self.residual:
            residual_weight = None
            if self.do_train:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                calc_device = (
                    torch.device(f"cuda:{local_rank}")
                    if torch.cuda.is_available()
                    else self.weight.device
                )

                U_g = self._group_quantize(U, self.U_scale).to(calc_device)
                V_g = self._group_quantize(V, self.V_scale).to(calc_device)
                approx = U_g @ V_g
                residual_weight = self.weight.data.float() - approx.to(
                    self.weight.device
                )
                del U_g, V_g, approx

            U_R, V_R = self._decompose_matrix(residual_weight)
            self.U_R = create_param(U_R)
            self.V_R = create_param(V_R)
            self.U_R_scale = create_param(self._init_group_scale(U_R))
            self.V_R_scale = create_param(self._init_group_scale(V_R))

        self.register_parameter("weight", None)
        self._quantized = False
        self._binarized = False

    def _decompose_matrix(self, X=None, target_split_dim=None):
        if target_split_dim is None:
            target_split_dim = self.split_dim

        if not self.do_train:
            U = torch.empty(self.out_features, target_split_dim)
            V = torch.empty(target_split_dim, self.in_features)
            return U, V

        X_f = X.float()
        U_t, S_t, Vh_t = torch.linalg.svd(X_f, full_matrices=False)

        S_sqrt_vec = torch.sqrt(S_t[:target_split_dim])
        S_sqrt_mat = torch.diag(S_sqrt_vec)

        U_temp = U_t[:, :target_split_dim] @ S_sqrt_mat
        V_temp = S_sqrt_mat @ Vh_t[:target_split_dim, :]

        with torch.no_grad():
            X_combined = torch.cat([U_temp, V_temp.t()], dim=0)
            R = self._compute_itq_rotation(X_combined, n_iter=50)
            U = (U_temp @ R).contiguous()
            V = (R.t() @ V_temp).contiguous()

        dtype = torch.bfloat16
        return U.to(dtype), V.to(dtype)

    def _num_groups(self, cols: int) -> int:
        return ceil(cols / self.group_size)

    def _reshape_groups(self, tensor):
        rows, cols = tensor.shape
        n_groups = self._num_groups(cols)
        padded_cols = n_groups * self.group_size
        if padded_cols != cols:
            pad = torch.zeros(
                rows,
                padded_cols - cols,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor.reshape(rows, n_groups, self.group_size)

    def _expand_group_scale(self, scale, cols: int):
        return scale.clamp_min(1e-6).repeat_interleave(self.group_size, dim=1)[
            :, :cols
        ]

    def _init_group_scale(self, tensor, n_iter: int = 5):
        with torch.no_grad():
            grouped = self._reshape_groups(tensor.float())
            scale = (grouped.abs().amax(dim=2, keepdim=True) / 2).clamp_min(1e-6)
            for _ in range(n_iter):
                q = torch.round(grouped / scale).clamp(
                    self.INT2_QUANT_MIN, self.INT2_QUANT_MAX
                )
                denominator = (q * q).sum(dim=2, keepdim=True)
                next_scale = (grouped * q).sum(
                    dim=2, keepdim=True
                ) / denominator.clamp_min(1e-6)
                scale = torch.where(
                    denominator > 0,
                    next_scale.clamp_min(1e-6),
                    scale,
                )
            return scale.squeeze(2).to(tensor.dtype)

    def _group_quantize(self, tensor, scale):
        dtype = tensor.dtype
        tensor_f = tensor.float()
        expanded_scale = self._expand_group_scale(scale.float(), tensor.shape[1])
        if self._quantized:
            return (tensor_f * expanded_scale).to(dtype)

        scaled = tensor_f / expanded_scale
        clipped = scaled.clamp(self.INT2_QUANT_MIN, self.INT2_QUANT_MAX)
        rounded = torch.round(clipped)
        q = clipped + (rounded - clipped).detach()
        return (q * expanded_scale).to(dtype)

    def _quantize_grouped_to_int(self, tensor, scale):
        expanded_scale = self._expand_group_scale(scale.float(), tensor.shape[1])
        q = torch.round(tensor.float() / expanded_scale).clamp(
            self.INT2_QUANT_MIN, self.INT2_QUANT_MAX
        )
        return q.to(torch.int8)

    def _dequantize_grouped_int(self, tensor, scale):
        expanded_scale = self._expand_group_scale(scale.float(), tensor.shape[1])
        return tensor.float() * expanded_scale

    def set_packed_mode(self, enabled: bool, *, do_train: bool = False):
        if enabled and do_train:
            with torch.no_grad():
                self.U.data = self._dequantize_grouped_int(
                    self.U.data, self.U_scale.data
                ).to(self.U.dtype)
                self.V.data = self._dequantize_grouped_int(
                    self.V.data, self.V_scale.data
                ).to(self.V.dtype)
                if self.residual:
                    self.U_R.data = self._dequantize_grouped_int(
                        self.U_R.data, self.U_R_scale.data
                    ).to(self.U_R.dtype)
                    self.V_R.data = self._dequantize_grouped_int(
                        self.V_R.data, self.V_R_scale.data
                    ).to(self.V_R.dtype)
            self._quantized = False
            self._binarized = False
            return

        self._quantized = enabled
        self._binarized = enabled

    def pack_weights(self):
        packed = {}

        def pack_param(param, scale, name):
            param_int = self._quantize_grouped_to_int(param.data, scale.data)
            packed[f"{name}_packed"] = int2_packer(
                param_int, quant_min=self.INT2_QUANT_MIN
            )
            packed[f"{name}_shape"] = torch.tensor(param.shape, dtype=torch.long)
            packed[f"{name}_bit_width"] = torch.tensor(2, dtype=torch.long)
            packed[f"{name}_quant_min"] = torch.tensor(
                self.INT2_QUANT_MIN, dtype=torch.long
            )

        pack_param(self.U, self.U_scale, "U")
        pack_param(self.V, self.V_scale, "V")
        if self.residual:
            pack_param(self.U_R, self.U_R_scale, "U_R")
            pack_param(self.V_R, self.V_R_scale, "V_R")
        return packed

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.get("prefix", "")
        state = nn.Module.state_dict(self, *args, **kwargs)
        for name in ("U", "V", "U_R", "V_R"):
            state.pop(prefix + name, None)

        packed_weights = self.pack_weights()
        for key, value in packed_weights.items():
            state[prefix + key] = value
        return state

    @property
    def group_size_used(self):
        return int(self._group_size.item())
