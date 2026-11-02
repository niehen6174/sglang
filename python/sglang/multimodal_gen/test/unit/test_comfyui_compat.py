# SPDX-License-Identifier: Apache-2.0
"""Community-workflow compatibility: speed-only patch nodes, custom LoRA loaders,
third-party LoRA stacks and several dynamic LoRA adapters on one layer."""

import sys
import types
from unittest import mock

import pytest
import torch

# Initialize the quantization registry before importing the LoRA layer modules.
import sglang.multimodal_gen.runtime.layers.quantization  # noqa: F401
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.graph_rewrite import (
    drop_speed_patches,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.lora_bridge import (
    add_sgld_lora,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA


def _graph(loader: str) -> dict:
    return {
        "1": {"class_type": loader, "inputs": {"unet_name": "dit.safetensors"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0]}},
        "3": {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {"model": ["2", 0], "lora_name": "turbo.safetensors", "strength": 0.8},
        },
        "4": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["3", 0]}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["4", 0]}},
    }


def test_speed_patches_on_sgld_models_are_dropped_and_custom_loras_routed() -> None:
    prompt = _graph("SGLDUNETLoader")
    changed = drop_speed_patches(prompt)
    assert sorted(changed) == [
        "2:ModelAttentionBackend",
        "3:MiniMaxH3TurboLoRA->SGLDLoraLoader",
        "4:PathchSageAttentionKJ",
    ]
    assert prompt["3"] == {
        "class_type": "SGLDLoraLoader",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "turbo.safetensors",
            "strength_model": 0.8,
            "nickname": "",
            "target": "all",
        },
    }
    assert prompt["5"]["inputs"]["model"] == ["3", 0]


def test_stock_models_keep_their_patch_nodes() -> None:
    prompt = _graph("UNETLoader")
    assert drop_speed_patches(prompt) == []
    assert prompt == _graph("UNETLoader")


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
    again = add_sgld_lora(add_sgld_lora(_Patcher(), "/loras/a.safetensors", 1.0), "/loras/b.safetensors", 0.5)
    assert again.model_options["sgld_lora_input"] == lora_input


def test_third_party_lora_nodes_reach_the_worker(tmp_path) -> None:
    """Every LoRA node ends in comfy.sd.load_lora_for_models; for an SGLD model
    the model half becomes a worker adapter and only CLIP is patched in ComfyUI."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core import lora_bridge

    calls = []
    comfy = types.ModuleType("comfy")
    comfy.sd = types.ModuleType("comfy.sd")
    comfy.utils = types.ModuleType("comfy.utils")
    comfy.sd.load_lora_for_models = lambda model, clip, lora, sm, sc, meta=None: (
        calls.append((model, clip, sm, sc)) or (model, "patched_clip")
    )
    comfy.utils.load_torch_file = lambda path, *a, **k: {"lora_unet_x.lora_down.weight": torch.zeros(1)}
    patcher_module = types.ModuleType("model_patcher")
    patcher_module.SGLDModelPatcher = _Patcher
    with mock.patch.dict(
        sys.modules,
        {
            "comfy": comfy,
            "comfy.sd": comfy.sd,
            "comfy.utils": comfy.utils,
            "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher": patcher_module,
        },
    ), mock.patch.object(lora_bridge, "_installed", False):
        lora_bridge.install()
        lora = comfy.utils.load_torch_file("/loras/style.safetensors", safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(_Patcher(), "clip", lora, 0.7, 1.0)
        stock_model, _ = comfy.sd.load_lora_for_models("stock", None, lora, 1.0, 0)
    assert model.model_options["sgld_lora_input"]["lora_path"] == ["/loras/style.safetensors"]
    assert model.model_options["sgld_lora_input"]["strength"] == [0.7]
    assert clip == "patched_clip"
    # CLIP-only pass for the SGLD model, untouched call for the stock one.
    assert calls == [(None, "clip", 0, 1.0), ("stock", None, 1.0, 0)]
    assert stock_model == "stock"


def test_dynamic_adapters_on_one_layer_sum_their_updates() -> None:
    """Quantized layers cannot merge LoRAs; several dynamic adapters are stacked
    along the rank axis instead of being rejected."""
    torch.manual_seed(0)
    base = torch.nn.Linear(32, 16, bias=False)
    layer = LinearWithLoRA(base)
    x = torch.randn(3, 32)
    expected = base(x)
    for i, (rank, alpha, strength) in enumerate([(4, 4, 1.0), (8, 16, 0.5), (2, 1, 0.7)]):
        A, B = torch.randn(rank, 32) * 0.1, torch.randn(16, rank) * 0.1
        expected = expected + strength * (alpha / rank) * (x @ A.T @ B.T)
        layer.lora_rank, layer.lora_alpha = rank, alpha
        layer.set_lora_weights(A, B, strength=strength, merge_weights=False, clear_existing=i == 0)
    out = layer(x)
    out = out[0] if isinstance(out, tuple) else out
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    assert len(layer.lora_weights_list) == 3
