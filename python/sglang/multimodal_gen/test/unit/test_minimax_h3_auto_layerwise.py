# SPDX-License-Identifier: Apache-2.0
"""Auto layerwise offload policy tests for non-Wan deployment configs."""

from unittest import mock

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3ComfyQuant4090PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.runtime.server_args.auto_tune import ServerArgsAutoTuner


def _tuner_for(config, *, performance_mode="auto"):
    server_args = mock.Mock()
    server_args.pipeline_config = config
    server_args.performance_mode = performance_mode
    server_args.dmd_denoising_steps = None
    server_args.use_fsdp_inference = False
    server_args.is_arg_explicitly_set = mock.Mock(return_value=False)
    return ServerArgsAutoTuner(server_args)


def test_h3_comfy_profile_auto_enables_dit_layerwise_offload():
    tuner = _tuner_for(MiniMaxH3ComfyQuant4090PipelineConfig(), performance_mode="memory")
    with mock.patch(
        "sglang.multimodal_gen.runtime.server_args.auto_tune.envs.SGLANG_CACHE_DIT_ENABLED",
        False,
    ):
        assert tuner._should_auto_enable_dit_layerwise_offload() is True


def test_wan_auto_mode_still_requires_a14b_for_layerwise():
    tuner = _tuner_for(WanT2V480PConfig(), performance_mode="auto")
    with mock.patch(
        "sglang.multimodal_gen.runtime.server_args.auto_tune.current_platform.enable_dit_layerwise_offload_for_wan_by_default",
        return_value=True,
    ), mock.patch(
        "sglang.multimodal_gen.runtime.server_args.auto_tune.envs.SGLANG_CACHE_DIT_ENABLED",
        False,
    ):
        assert tuner._should_auto_enable_dit_layerwise_offload() is False
