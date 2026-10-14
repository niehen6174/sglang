# SPDX-License-Identifier: Apache-2.0
"""FP32 indexed affine modulation with explicit eager rounding points."""

import torch
import triton
import triton.language as tl


@triton.jit
def _indexed_scale_shift_fp32_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    out_ptr,
    N_ELEMENTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SHIFT_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N_ELEMENTS
    index = tl.load(indices_ptr + off // HIDDEN, mask=mask, other=0)
    col = off % HIDDEN
    x = tl.load(x_ptr + off, mask=mask, other=0).to(tl.float32)
    shift = tl.load(shift_ptr + index * SHIFT_STRIDE + col, mask=mask, other=0).to(
        tl.float32
    )
    scale = tl.load(scale_ptr + index * SCALE_STRIDE + col, mask=mask, other=0).to(
        tl.float32
    )
    value = tl.inline_asm_elementwise(
        "{ .reg .f32 q,p; add.rn.f32 q, $3, 0f3f800000; mul.rn.f32 p, $1, q; add.rn.f32 $0, p, $2; }",
        constraints="=f,f,f,f",
        args=[x, shift, scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_ptr + off, value, mask=mask)


def indexed_scale_shift_fp32(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    if x.numel():
        _indexed_scale_shift_fp32_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            shift,
            scale,
            indices,
            out,
            x.numel(),
            x.shape[-1],
            shift.stride(0),
            scale.stride(0),
            1024,
            enable_fp_fusion=False,
        )
    return out
