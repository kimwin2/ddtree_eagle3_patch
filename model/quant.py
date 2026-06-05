"""Fake (simulated) activation quantization for the DFlash draft model.

This implements *fake* quantization (quant -> dequant in floating point) so that
only the quantization error is injected into the activations, matching what a
QNN / QAIRT HTP A16 deployment would do numerically without producing real
integer tensors.

Scheme (per the deployment target):
    * Activation only (weights are left untouched here).
    * uint16, asymmetric, per-tensor.
    * Calibration: min-max by default, percentile (histogram) when outliers are
      severe.

All quantizers default to *disabled* (identity), so attaching them to a model
does not change its numerics until calibration explicitly enables them. The
observer/quant state lives in non-persistent buffers, so it never interferes
with ``from_pretrained`` / ``save_pretrained`` weight loading.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


# Maximum number of elements fed to torch.quantile / histogram per call. Larger
# tensors are uniformly subsampled to keep calibration cheap.
_MAX_OBSERVE_ELEMS = 1 << 22


def _flatten_finite(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().reshape(-1).float()
    finite = torch.isfinite(flat)
    if not bool(finite.all()):
        flat = flat[finite]
    return flat


def _maybe_subsample(flat: torch.Tensor, limit: int = _MAX_OBSERVE_ELEMS) -> torch.Tensor:
    if flat.numel() <= limit:
        return flat
    stride = (flat.numel() + limit - 1) // limit
    return flat[::stride]


class MinMaxObserver(nn.Module):
    """Tracks the running global min/max of observed activations."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float("inf")), persistent=False)
        self.register_buffer("max_val", torch.tensor(float("-inf")), persistent=False)

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        flat = _flatten_finite(x)
        if flat.numel() == 0:
            return
        cur_min = flat.min()
        cur_max = flat.max()
        self.min_val = torch.minimum(self.min_val.to(cur_min.device), cur_min)
        self.max_val = torch.maximum(self.max_val.to(cur_max.device), cur_max)

    def has_stats(self) -> bool:
        return bool(torch.isfinite(self.min_val)) and bool(torch.isfinite(self.max_val))

    def compute_min_max(self) -> tuple[float, float]:
        return float(self.min_val.item()), float(self.max_val.item())

    def raw_min_max(self) -> tuple[float, float]:
        return float(self.min_val.item()), float(self.max_val.item())

    def reset(self) -> None:
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))


class HistogramObserver(nn.Module):
    """Histogram-based percentile observer.

    Accumulates a histogram over an adaptively growing range (ORT-style range
    merge) and clips the calibration range at the requested lower/upper
    percentiles. Use this when activations have heavy outliers that would
    otherwise blow up the min-max range.
    """

    def __init__(self, num_bins: int = 2048, percentile: float = 99.99) -> None:
        super().__init__()
        if not (50.0 < percentile <= 100.0):
            raise ValueError(f"percentile must be in (50, 100], got {percentile}")
        self.num_bins = int(num_bins)
        self.percentile = float(percentile)
        self.register_buffer("histogram", torch.zeros(self.num_bins, dtype=torch.float64), persistent=False)
        self.register_buffer("range_min", torch.tensor(float("inf"), dtype=torch.float64), persistent=False)
        self.register_buffer("range_max", torch.tensor(float("-inf"), dtype=torch.float64), persistent=False)

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        flat = _maybe_subsample(_flatten_finite(x)).double().cpu()
        if flat.numel() == 0:
            return
        cur_min = float(flat.min().item())
        cur_max = float(flat.max().item())
        if not bool(torch.isfinite(self.range_min)):
            self._init_range(cur_min, cur_max)
        else:
            self._expand_range(cur_min, cur_max)
        lo = float(self.range_min.item())
        hi = float(self.range_max.item())
        hist = torch.histc(flat, bins=self.num_bins, min=lo, max=hi)
        self.histogram += hist.double()

    def _init_range(self, lo: float, hi: float) -> None:
        if hi <= lo:
            hi = lo + 1e-8
        self.range_min.fill_(lo)
        self.range_max.fill_(hi)
        self.histogram.zero_()

    def _expand_range(self, cur_min: float, cur_max: float) -> None:
        lo = float(self.range_min.item())
        hi = float(self.range_max.item())
        new_lo = min(lo, cur_min)
        new_hi = max(hi, cur_max)
        if new_lo == lo and new_hi == hi:
            return
        # Re-bin the existing histogram into the widened range so accumulated
        # counts stay consistent with the new bin edges.
        if new_hi <= new_lo:
            new_hi = new_lo + 1e-8
        old_edges = torch.linspace(lo, hi, self.num_bins + 1, dtype=torch.float64)
        old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])
        new_width = (new_hi - new_lo) / self.num_bins
        target_bins = torch.clamp(
            ((old_centers - new_lo) / new_width).floor().long(),
            0,
            self.num_bins - 1,
        )
        rebinned = torch.zeros(self.num_bins, dtype=torch.float64)
        rebinned.index_add_(0, target_bins, self.histogram)
        self.histogram = rebinned
        self.range_min.fill_(new_lo)
        self.range_max.fill_(new_hi)

    def has_stats(self) -> bool:
        return bool(torch.isfinite(self.range_min)) and float(self.histogram.sum().item()) > 0

    def compute_min_max(self) -> tuple[float, float]:
        lo = float(self.range_min.item())
        hi = float(self.range_max.item())
        total = float(self.histogram.sum().item())
        if total <= 0:
            return lo, hi
        edges = torch.linspace(lo, hi, self.num_bins + 1, dtype=torch.float64)
        cdf = torch.cumsum(self.histogram, dim=0) / total
        lower_q = (1.0 - self.percentile / 100.0)
        upper_q = self.percentile / 100.0
        lower_idx = int(torch.searchsorted(cdf, torch.tensor(lower_q, dtype=torch.float64)).item())
        upper_idx = int(torch.searchsorted(cdf, torch.tensor(upper_q, dtype=torch.float64)).item())
        lower_idx = min(max(lower_idx, 0), self.num_bins - 1)
        upper_idx = min(max(upper_idx, 0), self.num_bins - 1)
        clipped_min = float(edges[lower_idx].item())
        clipped_max = float(edges[upper_idx + 1].item())
        if clipped_max <= clipped_min:
            clipped_max = clipped_min + 1e-8
        return clipped_min, clipped_max

    def raw_min_max(self) -> tuple[float, float]:
        return float(self.range_min.item()), float(self.range_max.item())

    def reset(self) -> None:
        self.histogram.zero_()
        self.range_min.fill_(float("inf"))
        self.range_max.fill_(float("-inf"))


class FakeQuantize(nn.Module):
    """Per-tensor asymmetric uint16 fake-quantizer with a built-in observer.

    Modes (mutually exclusive precedence: observe wins during calibration):
        * ``observing=True``  -> record activation statistics, pass input through.
        * ``enabled=True``    -> apply quant->dequant using calibrated qparams.
        * otherwise           -> identity.
    """

    def __init__(
        self,
        num_bits: int = 16,
        observer: str = "minmax",
        percentile: float = 99.99,
        num_bins: int = 2048,
    ) -> None:
        super().__init__()
        self.num_bits = int(num_bits)
        self.qmin = 0
        self.qmax = (1 << self.num_bits) - 1  # unsigned -> uint16: [0, 65535]
        self.observer_kind = observer
        if observer == "minmax":
            self.observer: nn.Module = MinMaxObserver()
        elif observer == "percentile":
            self.observer = HistogramObserver(num_bins=num_bins, percentile=percentile)
        else:
            raise ValueError(f"Unknown observer '{observer}' (expected 'minmax' or 'percentile')")

        self.enabled = False
        self.observing = False
        self.register_buffer("scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("zero_point", torch.tensor(0.0), persistent=False)
        self.register_buffer("calibrated", torch.tensor(False), persistent=False)

    def reconfigure(
        self,
        num_bits: int = 16,
        observer: str = "minmax",
        percentile: float = 99.99,
        num_bins: int = 2048,
    ) -> None:
        """Rebuild the observer / bit-width in place (before calibration)."""
        self.num_bits = int(num_bits)
        self.qmin = 0
        self.qmax = (1 << self.num_bits) - 1
        self.observer_kind = observer
        if observer == "minmax":
            self.observer = MinMaxObserver()
        elif observer == "percentile":
            self.observer = HistogramObserver(num_bins=num_bins, percentile=percentile)
        else:
            raise ValueError(f"Unknown observer '{observer}' (expected 'minmax' or 'percentile')")
        self.calibrated = torch.tensor(False)

    @torch.no_grad()
    def compute_qparams(self) -> None:
        if not self.observer.has_stats():
            return
        min_val, max_val = self.observer.compute_min_max()
        # Asymmetric range; ensure it is non-degenerate.
        if max_val <= min_val:
            max_val = min_val + 1e-8
        scale = (max_val - min_val) / float(self.qmax - self.qmin)
        scale = max(scale, 1e-12)
        zero_point = round(self.qmin - min_val / scale)
        zero_point = min(max(zero_point, self.qmin), self.qmax)
        self.scale = torch.tensor(float(scale))
        self.zero_point = torch.tensor(float(zero_point))
        self.calibrated = torch.tensor(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observing:
            self.observer.observe(x)
            return x
        if not self.enabled or not bool(self.calibrated):
            return x
        scale = self.scale.to(device=x.device, dtype=torch.float32)
        zero_point = self.zero_point.to(device=x.device, dtype=torch.float32)
        xf = x.float()
        q = torch.clamp(torch.round(xf / scale + zero_point), self.qmin, self.qmax)
        dq = (q - zero_point) * scale
        return dq.to(dtype=x.dtype)

    def export_state(self) -> dict:
        return {
            "num_bits": self.num_bits,
            "observer_kind": self.observer_kind,
            "scale": float(self.scale.item()),
            "zero_point": float(self.zero_point.item()),
            "calibrated": bool(self.calibrated),
        }

    def load_state(self, state: dict) -> None:
        self.scale = torch.tensor(float(state["scale"]))
        self.zero_point = torch.tensor(float(state["zero_point"]))
        self.calibrated = torch.tensor(bool(state["calibrated"]))


# --------------------------------------------------------------------------- #
# Model-wide helpers
# --------------------------------------------------------------------------- #

def iter_fake_quants(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantize):
            yield name, module


def set_observing(model: nn.Module, observing: bool) -> None:
    for _, fq in iter_fake_quants(model):
        fq.observing = observing


def set_enabled(model: nn.Module, enabled: bool) -> None:
    for _, fq in iter_fake_quants(model):
        fq.enabled = enabled


def reset_observers(model: nn.Module) -> None:
    for _, fq in iter_fake_quants(model):
        fq.observer.reset()
        fq.calibrated = torch.tensor(False)


def compute_all_qparams(model: nn.Module) -> int:
    count = 0
    for _, fq in iter_fake_quants(model):
        fq.compute_qparams()
        if bool(fq.calibrated):
            count += 1
    return count


def export_qparams(model: nn.Module) -> dict:
    return {name: fq.export_state() for name, fq in iter_fake_quants(model)}


def load_qparams(model: nn.Module, state: dict) -> int:
    loaded = 0
    for name, fq in iter_fake_quants(model):
        if name in state:
            fq.load_state(state[name])
            loaded += 1
    return loaded


def configure_fake_quants(model: nn.Module, quant_config: Optional[dict]) -> int:
    """Reconfigure every FakeQuantize in ``model`` from a config dict."""
    config = quant_config or {}
    count = 0
    for _, fq in iter_fake_quants(model):
        fq.reconfigure(
            num_bits=config.get("num_bits", 16),
            observer=config.get("observer", "minmax"),
            percentile=config.get("percentile", 99.99),
            num_bins=config.get("num_bins", 2048),
        )
        count += 1
    return count


def quant_summary(model: nn.Module) -> list[dict]:
    """Return per-quantizer observed min/max and calibrated qparams."""
    rows: list[dict] = []
    for name, fq in iter_fake_quants(model):
        observer = fq.observer
        if hasattr(observer, "raw_min_max"):
            raw_min, raw_max = observer.raw_min_max()
        else:
            raw_min = raw_max = float("nan")
        if observer.has_stats():
            calib_min, calib_max = observer.compute_min_max()
        else:
            calib_min = calib_max = float("nan")
        rows.append(
            {
                "name": name,
                "observer": fq.observer_kind,
                "min": raw_min,
                "max": raw_max,
                "calib_min": calib_min,
                "calib_max": calib_max,
                "scale": float(fq.scale.item()),
                "zero_point": float(fq.zero_point.item()),
                "calibrated": bool(fq.calibrated),
            }
        )
    return rows


def format_quant_summary(model: nn.Module) -> str:
    """Human-readable table of every quantizer's min/max and qparams."""
    rows = quant_summary(model)
    header = (
        f"{'layer (quantizer)':<48} {'obs':<10} "
        f"{'min':>14} {'max':>14} {'calib_min':>14} {'calib_max':>14} "
        f"{'scale':>14} {'zero_pt':>10} {'ok':>3}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['name']:<48} {row['observer']:<10} "
            f"{row['min']:>14.6g} {row['max']:>14.6g} "
            f"{row['calib_min']:>14.6g} {row['calib_max']:>14.6g} "
            f"{row['scale']:>14.6g} {row['zero_point']:>10.6g} "
            f"{('Y' if row['calibrated'] else 'n'):>3}"
        )
    calibrated = sum(1 for row in rows if row["calibrated"])
    lines.append("-" * len(header))
    lines.append(f"total quantizers={len(rows)} calibrated={calibrated}")
    return "\n".join(lines)


def make_fake_quant(quant_config: Optional[dict]) -> FakeQuantize:
    """Build a FakeQuantize from a config dict (or defaults)."""
    config = quant_config or {}
    return FakeQuantize(
        num_bits=config.get("num_bits", 16),
        observer=config.get("observer", "minmax"),
        percentile=config.get("percentile", 99.99),
        num_bins=config.get("num_bins", 2048),
    )
