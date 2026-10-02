"""Head-sharded Ulysses Q/K/V/gate slices match the replicated projection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    slice_head_rows,
    slice_merged_qkv_heads,
)


def test_slice_merged_qkv_matches_full_gemm():
    heads, head_dim, hidden, world, rows = 8, 4, 16, 4, 6
    x = torch.randn(rows, hidden)
    weight = torch.randn(3 * heads * head_dim, hidden)
    full = x @ weight.T
    local_heads = heads // world
    width = local_heads * head_dim
    for rank in range(world):
        sharded = slice_merged_qkv_heads(weight, heads, head_dim, rank, world)
        got = x @ sharded.T
        parts = []
        for section in range(3):
            base = section * heads * head_dim + rank * width
            parts.append(full[:, base : base + width])
        torch.testing.assert_close(got, torch.cat(parts, dim=-1))


def test_slice_gate_rows_match_full_gemm():
    heads, head_dim, hidden, world = 8, 4, 16, 4
    x = torch.randn(5, hidden)
    weight = torch.randn(heads * head_dim, hidden)
    full = x @ weight.T
    width = heads * head_dim // world
    for rank in range(world):
        got = x @ slice_head_rows(weight, heads, head_dim, rank, world).T
        torch.testing.assert_close(got, full[:, rank * width : (rank + 1) * width])


def _merged_lora_layer(monkeypatch, *, inner, tp_size, tp_rank, head_shard):
    """A MergedColumnParallelLinearWithLoRA over a stub qkv base layer."""
    from sglang.multimodal_gen.runtime.layers.lora import linear as lora_linear

    monkeypatch.setattr(lora_linear, "get_tp_rank", lambda: tp_rank)
    layer = lora_linear.MergedColumnParallelLinearWithLoRA.__new__(
        lora_linear.MergedColumnParallelLinearWithLoRA
    )
    layer.base_layer = SimpleNamespace(
        output_sizes=[inner] * 3,
        output_partition_sizes=[inner // tp_size] * 3,
        ulysses_head_shard=head_shard,
    )
    return layer


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_head_sharded_lora_b_matches_sharded_base_rows(monkeypatch, tp_size, tp_rank):
    """A LoRA delta on a head-sharded qkv must cover the rows the base kept.

    Adapters are loaded at the checkpoint's full head width, while Ulysses
    head-sharding narrows the base weight a second time inside the TP
    partition. If the two disagree the delta lands on the wrong heads.
    """
    heads, head_dim, hidden, world, rows, rank_r = 8, 4, 16, 4, 6, 3
    inner = heads * head_dim
    tp_heads = heads // tp_size
    x = torch.randn(rows, hidden)
    weight = torch.randn(3 * inner, hidden)
    lora_a = torch.randn(rank_r, hidden)
    lora_b = torch.randn(3 * inner, rank_r)

    full = x @ weight.T + (x @ lora_a.T) @ lora_b.T

    # The TP partition this rank owns, before any head-sharding.
    tp_width = tp_heads * head_dim
    tp_weight = torch.cat(
        [
            weight[section * inner + tp_rank * tp_width :][:tp_width]
            for section in range(3)
        ],
        dim=0,
    )

    for ulysses_rank in range(world):
        layer = _merged_lora_layer(
            monkeypatch,
            inner=inner,
            tp_size=tp_size,
            tp_rank=tp_rank,
            head_shard=(ulysses_rank, world),
        )
        sharded_weight = slice_merged_qkv_heads(
            tp_weight, tp_heads, head_dim, ulysses_rank, world
        )
        sharded_b = layer.slice_lora_b_weights(lora_b)
        got = x @ sharded_weight.T + (x @ lora_a.T) @ sharded_b.T

        width = tp_heads * head_dim // world
        expected = torch.cat(
            [
                full[
                    :,
                    section * inner
                    + tp_rank * tp_width
                    + ulysses_rank * width : section * inner
                    + tp_rank * tp_width
                    + (ulysses_rank + 1) * width,
                ]
                for section in range(3)
            ],
            dim=-1,
        )
        torch.testing.assert_close(got, expected)


def test_lora_b_slice_is_identity_without_head_shard(monkeypatch):
    """Layers that were never head-sharded keep the plain TP slice."""
    from sglang.multimodal_gen.runtime.layers.lora import linear as lora_linear

    inner, rank_r = 32, 3
    lora_b = torch.randn(3 * inner, rank_r)
    layer = _merged_lora_layer(
        monkeypatch, inner=inner, tp_size=1, tp_rank=0, head_shard=None
    )
    torch.testing.assert_close(layer.slice_lora_b_weights(lora_b), lora_b)
    assert lora_linear.BaseLayerWithLoRA._head_shard_window(layer, 128) == (0, 128)
