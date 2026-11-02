# SPDX-License-Identifier: Apache-2.0
"""Third-party ComfyUI LoRA nodes reach the SGLD worker."""

import sys
import types
from unittest import mock

import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.lora_bridge import (
    add_sgld_lora,
)


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
