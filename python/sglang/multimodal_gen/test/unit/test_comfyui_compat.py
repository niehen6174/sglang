# SPDX-License-Identifier: Apache-2.0
"""Community-workflow compatibility: speed-only patch nodes, custom LoRA loaders,
third-party LoRA stacks and several dynamic LoRA adapters on one layer."""

import json
import sys
import types
from unittest import mock

import pytest
import torch

# Initialize the quantization registry before importing the LoRA layer modules.
import sglang.multimodal_gen.runtime.layers.quantization  # noqa: F401
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core import graph_rewrite
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
            "inputs": {
                "model": ["2", 0],
                "lora_name": "turbo.safetensors",
                "strength": 0.8,
            },
        },
        "4": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["3", 0]}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["4", 0]}},
    }


@pytest.fixture
def backends(monkeypatch):
    """Attention backends this environment has; every file is a MiniMax-H3 DiT."""
    available = {"sage_attention2"}
    monkeypatch.setattr(graph_rewrite, "_available", lambda b: b in available)
    monkeypatch.setattr(graph_rewrite, "_model_type", lambda loader: "minimax_h3")
    return available


def test_speed_patches_on_sgld_models_are_dropped_and_custom_loras_routed(
    backends,
) -> None:
    prompt = _graph("SGLDUNETLoader")
    changed = drop_speed_patches(prompt)
    assert sorted(changed) == [
        "1:SGLDOptions(attention_backend=sage_attention2)",
        "2:ModelAttentionBackend",
        "3:MiniMaxH3TurboLoRA->SGLDLoraLoader",
        "4:PathchSageAttentionKJ",
    ]
    assert prompt["1"]["inputs"]["sgld_options"] == ["1.sgld_options", 0]
    assert prompt["1.sgld_options"] == {
        "class_type": "SGLDOptions",
        "inputs": {"attention_backend": "sage_attention2"},
    }
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


def test_speed_ups_carry_over_unless_the_user_chose_a_value(
    backends, monkeypatch
) -> None:
    """Sol falls back to Sage when sol-attn is missing; caches map to cache-dit;
    a value the user set on SGLDOptions wins; a setting the model cannot run is
    not carried (FastH3 keeps its VSA default, Qwen-Image 2.1 has no step cache)."""
    prompt = {
        "1": {
            "class_type": "SGLDUNETLoader",
            "inputs": {"unet_name": "h3.safetensors", "sgld_options": ["9", 0]},
        },
        "9": {"class_type": "SGLDOptions", "inputs": {"attention_backend": "auto"}},
        "2": {"class_type": "SolAttnMiniMax", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "H3MiniMaxCache", "inputs": {"model": ["2", 0]}},
        "4": {"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["3", 0]}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["4", 0]}},
    }
    drop_speed_patches(prompt)
    assert prompt["9"]["inputs"] == {
        "attention_backend": "sage_attention2",
        "cache_mode": "cache_dit",
    }
    assert prompt["5"]["inputs"]["model"] == ["1", 0]

    backends.add("sol_attention")
    prompt = _graph("SGLDUNETLoader")
    prompt["2"] = {"class_type": "SolAttnMiniMax", "inputs": {"model": ["1", 0]}}
    drop_speed_patches(prompt)
    # The first attention patch on the chain wins, as in ComfyUI.
    assert prompt["1.sgld_options"]["inputs"]["attention_backend"] == "sol_attention"

    # Sol keeps the node's sparsity: every step sparse at its tau, not SGLD's
    # 10 dense warm-up steps.
    prompt = _graph("SGLDUNETLoader")
    prompt["2"] = {
        "class_type": "MiniMaxH3MemoryEfficientSolAttentionPatch",
        "inputs": {
            "model": ["1", 0],
            "tau": 1.3,
            "thresh_type": "diag",
            "dense_blocks": "",
        },
    }
    drop_speed_patches(prompt)
    assert json.loads(
        prompt["1.sgld_options"]["inputs"]["attention_backend_config"]
    ) == {
        "dense_layers": "",
        "dense_steps": 0,
        "tau": 1.3,
        "thresh_type": "diag",
    }

    prompt = _graph("SGLDUNETLoader")
    prompt["1"]["inputs"]["sgld_options"] = ["9", 0]
    prompt["9"] = {
        "class_type": "SGLDOptions",
        "inputs": {"attention_backend": "flash_attention"},
    }
    drop_speed_patches(prompt)
    assert prompt["9"]["inputs"] == {"attention_backend": "flash_attention"}

    monkeypatch.setattr(graph_rewrite, "_model_type", lambda loader: "fast_h3")
    prompt = _graph("SGLDUNETLoader")
    drop_speed_patches(prompt)
    assert "sgld_options" not in prompt["1"]["inputs"]

    monkeypatch.setattr(graph_rewrite, "_model_type", lambda loader: "qwen_image21")
    prompt = _graph("SGLDUNETLoader")
    prompt["2"] = {"class_type": "EasyCache", "inputs": {"model": ["1", 0]}}
    drop_speed_patches(prompt)
    assert prompt["1.sgld_options"]["inputs"] == {
        "attention_backend": "sage_attention2"
    }


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
    again = add_sgld_lora(
        add_sgld_lora(_Patcher(), "/loras/a.safetensors", 1.0),
        "/loras/b.safetensors",
        0.5,
    )
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
    comfy.utils.load_torch_file = lambda path, *a, **k: {
        "lora_unet_x.lora_down.weight": torch.zeros(1)
    }
    patcher_module = types.ModuleType("model_patcher")
    patcher_module.SGLDModelPatcher = _Patcher
    with (
        mock.patch.dict(
            sys.modules,
            {
                "comfy": comfy,
                "comfy.sd": comfy.sd,
                "comfy.utils": comfy.utils,
                "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher": patcher_module,
            },
        ),
        mock.patch.object(lora_bridge, "_installed", False),
    ):
        lora_bridge.install()
        lora = comfy.utils.load_torch_file("/loras/style.safetensors", safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(_Patcher(), "clip", lora, 0.7, 1.0)
        stock_model, _ = comfy.sd.load_lora_for_models("stock", None, lora, 1.0, 0)
    assert model.model_options["sgld_lora_input"]["lora_path"] == [
        "/loras/style.safetensors"
    ]
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


def test_patches_behind_a_switch_node_are_found(backends) -> None:
    """PlagueKind's step cache sits after a ComfySwitchNode, not on a model chain."""
    prompt = {
        "1": {
            "class_type": "SGLDUNETLoader",
            "inputs": {"unet_name": "h3.safetensors"},
        },
        "2": {"class_type": "H3SLAAttention", "inputs": {"model": ["1", 0]}},
        "3": {
            "class_type": "ComfySwitchNode",
            "inputs": {"on_false": ["2", 0], "on_true": ["1", 0], "switch": False},
        },
        "4": {"class_type": "H3MiniMaxCache", "inputs": {"model": ["3", 0]}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["4", 0]}},
    }
    drop_speed_patches(prompt)
    assert "4" not in prompt and prompt["5"]["inputs"]["model"] == ["3", 0]
    assert prompt["1.sgld_options"]["inputs"] == {
        "attention_backend": "sage_attention2",
        "cache_mode": "cache_dit",
    }
