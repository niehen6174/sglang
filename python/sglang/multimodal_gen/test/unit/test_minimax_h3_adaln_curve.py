# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for MiniMax H3 AdaLN curve-table interpolation."""

import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_curve import (
    interpolate_adaln_t_table,
)


def _comfyui_reference_interpolate(
    adaln_t_table: torch.Tensor,
    t_vals: torch.Tensor,
) -> torch.Tensor:
    """Reference copy of ComfyUI comfy/ldm/minimax/model.py curve lookup."""
    table = adaln_t_table
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


def test_adaln_curve_interpolation_matches_comfyui_reference():
    torch.manual_seed(0)
    table = torch.randn(1025, 8, dtype=torch.float32)
    t_vals = torch.linspace(-0.25, 1.25, steps=257, dtype=torch.float32)

    for device in ("cpu",):
        ref = _comfyui_reference_interpolate(table.to(device), t_vals.to(device))
        actual = interpolate_adaln_t_table(table.to(device), t_vals.to(device))
        torch.testing.assert_close(actual, ref, rtol=0, atol=1e-6)
        assert (actual.max() - actual.min()).abs() >= 0


def test_adaln_curve_interpolation_endpoint_clamps():
    table = torch.arange(1025 * 8, dtype=torch.float32).view(1025, 8)
    t_vals = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float32)

    out = interpolate_adaln_t_table(table, t_vals)
    torch.testing.assert_close(out[0], table[0], rtol=0, atol=0)
    torch.testing.assert_close(out[1], table[-1], rtol=0, atol=0)
    torch.testing.assert_close(out[2], table[0], rtol=0, atol=0)
    torch.testing.assert_close(out[3], table[-1], rtol=0, atol=0)
