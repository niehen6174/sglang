# SPDX-License-Identifier: Apache-2.0
"""Several dynamic (unmerged) LoRA adapters on one layer."""

import torch

# Initialize the quantization registry before importing the LoRA layer modules.
import sglang.multimodal_gen.runtime.layers.quantization  # noqa: F401
from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA


def test_dynamic_adapters_on_one_layer_sum_their_updates() -> None:
    """Quantized layers cannot merge LoRAs; several dynamic adapters are stacked
    along the rank axis instead of being rejected."""
    torch.manual_seed(0)
    base = torch.nn.Linear(32, 16, bias=False)
    layer = LinearWithLoRA(base)
    x = torch.randn(3, 32)
    expected = base(x)
    for i, (rank, alpha, strength) in enumerate(
        [(4, 4, 1.0), (8, 16, 0.5), (2, 1, 0.7)]
    ):
        A, B = torch.randn(rank, 32) * 0.1, torch.randn(16, rank) * 0.1
        expected = expected + strength * (alpha / rank) * (x @ A.T @ B.T)
        layer.lora_rank, layer.lora_alpha = rank, alpha
        layer.set_lora_weights(
            A, B, strength=strength, merge_weights=False, clear_existing=i == 0
        )
    out = layer(x)
    out = out[0] if isinstance(out, tuple) else out
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    assert len(layer.lora_weights_list) == 3
