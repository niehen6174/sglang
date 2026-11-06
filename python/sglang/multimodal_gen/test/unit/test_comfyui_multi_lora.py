# SPDX-License-Identifier: Apache-2.0
"""Several LoRAs on an SGLD model: chained loader nodes and dynamic adapters on one layer."""

import torch

# Initialize the quantization registry before importing the LoRA layer modules.
import sglang.multimodal_gen.runtime.layers.quantization  # noqa: F401
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.sgld_lora import (
    add_sgld_lora,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA


class _Patcher:
    def __init__(self):
        self.patches, self.model_options = {}, {}

    def clone(self):
        n = _Patcher()
        n.patches = dict(self.patches)
        n.model_options = dict(self.model_options)
        return n


def test_chained_loras_become_one_worker_request() -> None:
    model = add_sgld_lora(_Patcher(), "/loras/a.safetensors", 1.0)
    model = add_sgld_lora(model, "/loras/b.safetensors", 0.5)
    lora_input = model.model_options["sgld_lora_input"]
    assert lora_input["lora_path"] == ["/loras/a.safetensors", "/loras/b.safetensors"]
    assert lora_input["strength"] == [1.0, 0.5]
    assert len(set(lora_input["lora_nickname"])) == 2
    # Same chain, same nicknames: the worker keeps its loaded adapters.
    again = add_sgld_lora(
        add_sgld_lora(_Patcher(), "/loras/a.safetensors", 1.0),
        "/loras/b.safetensors",
        0.5,
    )
    assert again.model_options["sgld_lora_input"] == lora_input


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


def test_grouped_and_plain_dynamic_adapters_stack() -> None:
    """A fused-QKV LoRA ([groups, rank, in] / [groups, out, rank]) and a plain one
    on the same quantized layer (LiblibAI's turbo + detail LoRA stack)."""
    from sglang.multimodal_gen.runtime.layers.lora.linear import _compute_lora_delta

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 16, bias=False)
    layer = LinearWithLoRA(base)
    x = torch.randn(3, 32)
    grouped = (torch.randn(2, 4, 32) * 0.1, torch.randn(2, 8, 4) * 0.1)
    plain = (torch.randn(6, 32) * 0.1, torch.randn(16, 6) * 0.1)
    expected = (
        base(x) + _compute_lora_delta(x, *grouped) + 0.5 * (x @ plain[0].T @ plain[1].T)
    )
    for i, ((A, B), strength) in enumerate([(grouped, 1.0), (plain, 0.5)]):
        layer.lora_rank = layer.lora_alpha = A.shape[-2]
        layer.set_lora_weights(
            A, B, strength=strength, merge_weights=False, clear_existing=i == 0
        )
    out = layer(x)
    out = out[0] if isinstance(out, tuple) else out
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
