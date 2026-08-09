# SPDX-License-Identifier: Apache-2.0
"""Kernel backend selection for ComfyUI-style quantized linear layers."""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_NVFP4_E2M1_LUT_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def ensure_comfy_kitchen_nvfp4_lut_materialized() -> None:
    """Re-materialize comfy_kitchen's NVFP4 LUT if it was created on meta device."""

    try:
        import comfy_kitchen.backends.eager.quantization as qmod
    except ImportError:
        return

    if qmod.E2M1_LUT.device.type != "meta":
        return

    qmod.E2M1_LUT = torch.tensor(
        _NVFP4_E2M1_LUT_VALUES,
        device="cpu",
    ).unsqueeze(1)
    qmod.E2M1_LUT_CACHE.clear()


class ComfyQuantKernelBackend(str, Enum):
    COMFY_KITCHEN = "comfy_kitchen"
    SGL_KERNEL = "sgl_kernel"
    TORCH = "torch"


_COMFY_KITCHEN_AVAILABLE: bool | None = None
_SGL_INT8_AVAILABLE: bool | None = None


def _probe_comfy_kitchen() -> bool:
    global _COMFY_KITCHEN_AVAILABLE
    if _COMFY_KITCHEN_AVAILABLE is not None:
        return _COMFY_KITCHEN_AVAILABLE
    try:
        import comfy_kitchen  # noqa: F401

        _COMFY_KITCHEN_AVAILABLE = True
    except Exception:
        _COMFY_KITCHEN_AVAILABLE = False
    return _COMFY_KITCHEN_AVAILABLE


def _probe_sgl_int8() -> bool:
    global _SGL_INT8_AVAILABLE
    if _SGL_INT8_AVAILABLE is not None:
        return _SGL_INT8_AVAILABLE
    try:
        from sgl_kernel import int8_scaled_mm  # noqa: F401

        _SGL_INT8_AVAILABLE = True
    except Exception:
        _SGL_INT8_AVAILABLE = False
    return _SGL_INT8_AVAILABLE


def select_int8_kernel_backend(
    *,
    force_full_precision: bool = False,
) -> ComfyQuantKernelBackend:
    if force_full_precision:
        return ComfyQuantKernelBackend.TORCH
    if _probe_comfy_kitchen():
        return ComfyQuantKernelBackend.COMFY_KITCHEN
    if _probe_sgl_int8():
        return ComfyQuantKernelBackend.SGL_KERNEL
    return ComfyQuantKernelBackend.TORCH


_CONVROT_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _build_convrot_hadamard(
    size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the normalized regular orthogonal Hadamard matrix used by ConvRot."""
    cache_key = (size, str(device), dtype)
    cached = _CONVROT_HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(
            f"ConvRot Hadamard size must be a power of 4, got {size}"
        )

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4

    h_normalized = h / (size**0.5)
    _CONVROT_HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def _rotate_convrot_activation(x: torch.Tensor, groupsize: int) -> torch.Tensor:
    """Apply online ConvRot activation rotation (x @ H per group)."""
    if groupsize <= 0 or x.shape[-1] % groupsize != 0:
        return x
    orig_shape = x.shape
    features = orig_shape[-1]
    n_groups = features // groupsize
    x_grouped = x.reshape(-1, n_groups, groupsize)
    h = _build_convrot_hadamard(groupsize, x.device, x.dtype)
    x_rotated = torch.matmul(x_grouped, h)
    return x_rotated.reshape(orig_shape)


def _torch_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    convrot: bool,
    convrot_groupsize: int,
) -> torch.Tensor:
    activation = (
        _rotate_convrot_activation(x, convrot_groupsize) if convrot else x
    )
    weight_f = weight.to(dtype=activation.dtype)
    if weight_scale.ndim == 2 and weight_scale.shape[1] == 1:
        scale = weight_scale.squeeze(-1).to(dtype=activation.dtype)
    else:
        scale = weight_scale.reshape(-1).to(dtype=activation.dtype)
    if weight_f.shape[0] == scale.shape[0]:
        dequant = weight_f * scale.unsqueeze(-1)
        out = activation @ dequant.transpose(-1, -2)
    else:
        dequant = weight_f * scale.unsqueeze(0)
        out = activation @ dequant
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


def _sgl_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    convrot: bool,
    convrot_groupsize: int,
) -> torch.Tensor:
    from sgl_kernel import int8_scaled_mm

    activation = (
        _rotate_convrot_activation(x, convrot_groupsize) if convrot else x
    )
    x_2d = activation.reshape(-1, activation.shape[-1])
    m, _ = x_2d.shape
    x_q = torch.empty(m, x_2d.shape[1], dtype=torch.int8, device=x_2d.device)
    x_scale = torch.empty(m, dtype=torch.float32, device=x_2d.device)
    x_q, x_scale = _per_token_quant_int8(x_2d, x_q, x_scale)

    if weight_scale.ndim == 2 and weight_scale.shape[1] == 1:
        w_scale = weight_scale.squeeze(-1)
    else:
        w_scale = weight_scale.reshape(-1)
    wqt = weight.t()
    out = int8_scaled_mm(x_q, wqt, x_scale, w_scale, out_dtype=x_2d.dtype)
    out = out.reshape(*activation.shape[:-1], out.shape[-1])
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


def _per_token_quant_int8(
    x: torch.Tensor,
    out_q: torch.Tensor,
    out_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 127.0
    out_q.copy_((x / scale).round().clamp(-128, 127).to(torch.int8))
    out_scale.copy_(scale.squeeze(-1))
    return out_q, out_scale


def _comfy_kitchen_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    convrot: bool,
    convrot_groupsize: int,
) -> torch.Tensor:
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    # comfy_kitchen applies ConvRot rotation internally when convrot=True.
    activation = x
    if weight_scale.ndim == 2 and weight_scale.shape[1] == 1:
        scale = weight_scale.squeeze(-1)
    else:
        scale = weight_scale.reshape(-1)

    qweight_data = weight
    if qweight_data.shape[0] != scale.shape[0]:
        qweight_data = qweight_data.t()
    params = TensorWiseINT8Layout.Params(
        scale=scale.to(device=qweight_data.device, dtype=torch.float32),
        orig_dtype=activation.dtype,
        orig_shape=tuple(qweight_data.shape),
        is_weight=True,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
        transposed=False,
    )
    qweight = QuantizedTensor(
        qweight_data.contiguous(),
        "TensorWiseINT8Layout",
        params,
    )
    out = torch.nn.functional.linear(activation, qweight)
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


_INT8_KERNELS: dict[
    ComfyQuantKernelBackend,
    Callable[..., torch.Tensor],
] = {
    ComfyQuantKernelBackend.TORCH: _torch_int8_linear,
    ComfyQuantKernelBackend.SGL_KERNEL: _sgl_int8_linear,
    ComfyQuantKernelBackend.COMFY_KITCHEN: _comfy_kitchen_int8_linear,
}


def apply_int8_linear(
    *,
    backend: ComfyQuantKernelBackend,
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    convrot: bool,
    convrot_groupsize: int,
) -> torch.Tensor:
    try:
        kernel = _INT8_KERNELS[backend]
        return kernel(
            x,
            weight,
            weight_scale,
            bias,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
        )
    except Exception:
        if backend == ComfyQuantKernelBackend.TORCH:
            raise
        logger.warning(
            "ComfyQuant INT8 backend %s failed; falling back to torch dequant",
            backend.value,
            exc_info=True,
        )
        return _torch_int8_linear(
            x,
            weight,
            weight_scale,
            bias,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
        )


__all__ = [
    "ComfyQuantKernelBackend",
    "apply_int8_linear",
    "select_int8_kernel_backend",
]
