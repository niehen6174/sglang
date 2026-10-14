# SPDX-License-Identifier: Apache-2.0
"""CUDA checks for indexed FP32 gate rounding and residual ownership."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows,hidden", [(0, 5376), (17, 5376), (15488, 5376)])
@pytest.mark.parametrize("inplace", [False, True])
def test_indexed_fp32_gate_matches_eager(rows, hidden, inplace):
    from sglang.kernels.ops.diffusion.modulate.indexed_fp32_gate_triton import (
        indexed_gate_fp32,
    )

    torch.manual_seed(42)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    other = torch.randn_like(x)
    # AdaLN chunk views have a row stride larger than their visible width.
    gate = torch.randn(3, hidden * 6, device="cuda", dtype=torch.float32)
    gate = gate[:, hidden * 2 : hidden * 3]
    indices = torch.arange(rows, device="cuda") % 3
    if rows:
        # A real cancellation case where a contracted FMA changes the answer.
        x[0, 0] = -0.1484375
        other[0, 0] = 14.25
        gate[0, 0] = 0.01041664183139801
    original = x.clone()
    expected = (x + gate.index_select(0, indices) * other).to(torch.bfloat16)
    result = indexed_gate_fp32(x, gate, other, indices, allow_inplace=inplace)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if rows:
        assert (result.data_ptr() == x.data_ptr()) == inplace
    if not inplace:
        torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("rows,hidden", [(0, 5376), (17, 5376), (15488, 5376)])
def test_indexed_fp32_scale_shift_matches_eager(rows, hidden):
    from sglang.kernels.ops.diffusion.modulate.indexed_fp32_scale_shift_triton import (
        indexed_scale_shift_fp32,
    )

    torch.manual_seed(43)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    original = x.clone()
    modulation = torch.randn(3, hidden * 6, device="cuda", dtype=torch.float32)
    shift, scale = modulation[:, :hidden], modulation[:, hidden : hidden * 2]
    indices = torch.arange(rows, device="cuda") % 3
    if rows:
        scale[0, 0] = -0.9999999403953552
        shift[0, 0] = 1.234567e-7
    expected = (
        x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    ).to(x.dtype)
    actual = indexed_scale_shift_fp32(x, shift, scale, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    if rows:
        assert actual.data_ptr() != x.data_ptr()
