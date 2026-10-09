# SPDX-License-Identifier: Apache-2.0
"""A failed worker reply must surface its error, not an empty noise_pred."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


def test_worker_error_is_raised_not_unpacked():
    """Unpacking a failed reply replaced the worker's message with a misleading
    adapter TypeError about noise_pred being None."""
    ex = SGLDiffusionExecutor.__new__(SGLDiffusionExecutor)
    torch.nn.Module.__init__(ex)
    ex.adapter = FluxAdapter()
    ex.model_path = "/test-model"
    ex.session_id, ex._run_id, ex._sent_conds = "error-test", 0, set()
    ex.generator = SimpleNamespace(
        server_args=SimpleNamespace(attention_backend_config={}, enable_trace=False),
        _send_to_scheduler_and_wait_for_response=lambda requests: SimpleNamespace(
            noise_pred=None, error="index_copy_(): shape mismatch"
        ),
    )
    x, t, context, y = (
        torch.randn(1, 16, 8, 8),
        torch.full((1,), 0.5),
        torch.randn(1, 7, 32),
        torch.randn(1, 768),
    )
    with patch(
        "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base.SamplingParams.from_user_sampling_params_args",
        side_effect=lambda model_path, server_args, **kw: SamplingParams(**kw),
    ), patch(
        "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base.torch.Generator",
        side_effect=lambda device: object(),
    ):
        packed = ex.adapter.pack(x, t, context, y=y)
        with pytest.raises(RuntimeError, match="worker failed: index_copy_"):
            ex._execute_packed(packed, x, t)
