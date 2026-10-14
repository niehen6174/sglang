# SPDX-License-Identifier: Apache-2.0
"""Indexed FP32 gated residual with BF16 inputs and output.

Preserve the eager FP32 multiply, FP32 add, and final BF16 rounding.
Explicit PTX instructions protect cancellation cases from FMA contraction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _indexed_gate_fp32_kernel(
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    out_ptr,
    N_ELEMENTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N_ELEMENTS
    row = off // HIDDEN
    col = off % HIDDEN
    index = tl.load(indices_ptr + row, mask=mask, other=0)
    x = tl.load(x_ptr + off, mask=mask, other=0).to(tl.float32)
    g = tl.load(gate_ptr + index * GATE_STRIDE + col, mask=mask, other=0).to(tl.float32)
    y = tl.load(other_ptr + off, mask=mask, other=0).to(tl.float32)
    # Keep both rounding points explicit; cancellation cases distinguish this
    # eager operation from a single FMA in the worker compilation environment.
    val = tl.inline_asm_elementwise(
        "{ .reg .f32 p; mul.rn.f32 p, $2, $3; add.rn.f32 $0, $1, p; }",
        constraints="=f,f,f,f",
        args=[x, g, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_ptr + off, val, mask=mask)


def indexed_gate_fp32(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    allow_inplace: bool = False,
) -> torch.Tensor:
    if x.numel() == 0:
        return x if allow_inplace else torch.empty_like(x)
    out = x if allow_inplace else torch.empty_like(x)
    _indexed_gate_fp32_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        gate,
        other,
        indices,
        out,
        x.numel(),
        x.shape[-1],
        gate.stride(0),
        1024,
        enable_fp_fusion=False,
    )
    return out
