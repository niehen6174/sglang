# SPDX-License-Identifier: Apache-2.0
"""AdaLN curve-table interpolation for MiniMax H3 pruned checkpoints."""

from __future__ import annotations

import torch


def interpolate_adaln_t_table(
    adaln_t_table: torch.Tensor,
    t_vals: torch.Tensor,
) -> torch.Tensor:
    """Interpolate AdaLN inputs from a curve lookup table.

    Matches ComfyUI ``MiniMaxH3Model`` when ``use_adaln_curves`` is enabled:

    .. code-block:: python

        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    Args:
        adaln_t_table: ``[grid, dim]`` fp32 curve table.
        t_vals: ``[M]`` normalized timesteps in ``[0, 1]``.

    Returns:
        ``[M, dim]`` interpolated embeddings in the table dtype.
    """
    table = adaln_t_table
    if table.dim() != 2:
        raise ValueError(
            f"adaln_t_table must be rank-2, got shape {tuple(table.shape)}"
        )
    if table.shape[0] < 2:
        raise ValueError(
            f"adaln_t_table requires at least 2 grid rows, got {table.shape[0]}"
        )

    t = t_vals.to(device=table.device, dtype=torch.float32).view(-1)
    pos = t.clamp(0.0, 1.0) * float(table.shape[0] - 1)
    i0 = pos.floor().to(dtype=torch.long).clamp(max=table.shape[0] - 2)
    frac = (pos - i0.to(dtype=pos.dtype)).unsqueeze(1)
    return torch.lerp(table[i0], table[i0 + 1], frac)


__all__ = ["interpolate_adaln_t_table"]
