# SPDX-License-Identifier: Apache-2.0
"""ComfyUI must keep tracking the model when a temporary SGLD clone expires."""

import gc
import weakref

import pytest
import torch

model_management = pytest.importorskip("comfy.model_management")

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher import (
    SGLDModelPatcher,
)


@pytest.mark.parametrize("depth", [1, 3])
def test_expired_clone_returns_loaded_model_to_live_parent(depth):
    model = torch.nn.Linear(1, 1)
    device = torch.device("cpu")
    parent = SGLDModelPatcher(model, device, device, model_type="minimax_h3")
    child = parent
    for _ in range(depth):
        child = child.clone()
    tracked = model_management.LoadedModel(child)
    tracked.real_model = weakref.ref(model)
    assert not tracked.is_dead()

    del child
    gc.collect()

    assert not tracked.is_dead()
    assert tracked.model is parent
    assert tracked.real_model() is model
