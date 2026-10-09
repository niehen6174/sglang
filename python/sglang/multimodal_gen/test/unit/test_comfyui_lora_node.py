# SPDX-License-Identifier: Apache-2.0
"""SGLDLoraLoader: chained LoRA nodes must keep every LoRA active."""

import copy
import sys
import types
from types import SimpleNamespace
from unittest import mock


def _load_nodes():
    """Import the real nodes.py with the ComfyUI modules it needs stubbed."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_full_path = lambda folder, name: f"/{folder}/{name}"
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")
    comfy_api_input.VideoInput = type("VideoInput", (), {})
    comfy_api.input = comfy_api_input
    stubs = {
        "folder_paths": folder_paths,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api_input,
    }
    with mock.patch.dict(sys.modules, stubs):
        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import nodes
    return nodes


class _Model:
    """ModelPatcher stand-in: clone() copies patches like SGLDModelPatcher.clone."""

    def __init__(self, executor, patches=None):
        self.model = SimpleNamespace(diffusion_model=executor)
        self.patches = dict(patches or {})
        self.model_options = {}

    def clone(self):
        clone = _Model(self.model.diffusion_model, self.patches)
        clone.model_options = copy.deepcopy(self.model_options)
        return clone


def test_chained_lora_nodes_keep_every_lora() -> None:
    nodes = _load_nodes()
    calls = []
    executor = SimpleNamespace(set_lora=lambda **kwargs: calls.append(kwargs))
    base = _Model(executor)

    loader = nodes.SGLDLoraLoader()
    (first,) = loader.load_lora(base, "style.safetensors", 1.0, nickname="style")
    (second,) = loader.load_lora(first, "detail.safetensors", 0.8, nickname="detail")

    assert calls == []
    desired = second.model_options["sgld_lora_input"]
    assert desired["lora_nickname"] == ["style", "detail"]
    assert desired["strength"] == [1.0, 0.8]
    assert desired["lora_path"] == [
        "/loras/style.safetensors",
        "/loras/detail.safetensors",
    ]
    assert "style" not in base.patches  # the upstream model is not mutated
    assert set(second.patches) == {"style", "detail"}


def test_sgld_model_rejects_native_lora_on_the_served_dit():
    import pytest

    pytest.importorskip("comfy.model_patcher")
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher import (
        SGLDModelPatcher,
    )

    patcher = object.__new__(SGLDModelPatcher)
    # A text-encoder LoRA through LoraLoader(model, clip) touches no DiT key.
    assert SGLDModelPatcher.add_patches(patcher, {"clip_l.layer.weight": ()}, 1.0) == []
    with pytest.raises(RuntimeError, match="SGLDLoraLoader"):
        SGLDModelPatcher.add_patches(
            patcher, {("diffusion_model.proj_out.weight", None): ()}, 0.5
        )


def test_executor_state_dict_lists_dit_keys_without_recursing():
    import torch

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    executor = SGLDiffusionExecutor.__new__(SGLDiffusionExecutor)
    torch.nn.Module.__init__(executor)
    executor.dit_state_keys = (
        "proj_out.weight",
        "transformer_blocks.0.ff.net.2.weight",
    )
    comfy_model = torch.nn.Module()
    comfy_model.diffusion_model = executor
    # The executor points back at the ComfyUI model without registering it.
    object.__setattr__(executor, "model", comfy_model)
    assert sorted(comfy_model.state_dict()) == [
        "diffusion_model.proj_out.weight",
        "diffusion_model.transformer_blocks.0.ff.net.2.weight",
    ]


def test_evict_comfy_models_keeps_only_the_sampled_sgld_model(monkeypatch):
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        evict_comfy_models,
    )

    sgld, text_encoder = object(), object()
    loaded = [types.SimpleNamespace(model=m) for m in (sgld, text_encoder)]
    calls = []
    mm = types.SimpleNamespace(
        current_loaded_models=loaded,
        get_torch_device=lambda: "cuda:0",
        free_memory=lambda n, dev, keep_loaded=(): calls.append((n, keep_loaded)),
        soft_empty_cache=lambda: None,
    )
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    evict_comfy_models(keep=sgld)
    ((required, keep),) = calls
    # Ask for more than any device has, so every other model unloads fully.
    assert required >= 1e30
    assert [entry.model for entry in keep] == [sgld]


def test_sampler_run_evicts_comfy_models_but_keeps_the_sampled_model(monkeypatch):
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors import base

    calls = []
    monkeypatch.setattr(
        base, "evict_comfy_models", lambda keep=None: calls.append(keep)
    )
    executor = base.SGLDiffusionExecutor.__new__(base.SGLDiffusionExecutor)
    executor._ensure_runtime = None
    executor._lora_input = None
    executor._run_id = 0
    patcher = SimpleNamespace(model_options={})
    out = executor.sampler_sample_wrapper(
        lambda *a, **k: "sampled", SimpleNamespace(model_patcher=patcher)
    )
    assert out == "sampled"
    assert calls == [patcher]
