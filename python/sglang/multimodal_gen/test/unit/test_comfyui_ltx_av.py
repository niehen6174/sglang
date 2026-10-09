# SPDX-License-Identifier: Apache-2.0
"""LTX-2.x audio-video ComfyUI integrated mode: checkpoint spec and step stage."""

from __future__ import annotations

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.ltx_2_5 import LTX25Config

TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "cross_attention_dim": 32,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 8,
    "audio_cross_attention_dim": 16,
    "caption_channels": 32,
    "in_channels": 16,
    "out_channels": 16,
    "rope_type": "split",
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quantization_ignored_layers_match_ltx2_module_paths(
    single_process_model_parallel,
):
    """--quantization-ignored-layers must reach LTX-2 linears by module path."""
    from sglang.multimodal_gen.runtime.layers.linear import LinearBase
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
    from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
        LTX2VideoTransformer3DModel,
    )

    config = LTX25Config()
    for key, value in {**TINY, "num_layers": 1}.items():
        if key != "rope_type":
            setattr(config.arch_config, key, value)
    config.arch_config.__post_init__()
    quant_config = Fp8Config(ignored_layers=["transformer_blocks.0.attn1", "proj_out"])
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel(
            config=config, hf_config={}, quant_config=quant_config
        )
    methods = {
        name: type(module.quant_method).__name__
        for name, module in model.named_modules()
        if isinstance(module, LinearBase)
    }
    for name in ("transformer_blocks.0.attn1.to_q", "proj_out"):
        assert methods[name] == "UnquantizedLinearMethod", name
    for name in ("transformer_blocks.0.attn2.to_q", "patchify_proj"):
        assert methods[name] == "Fp8LinearMethod", name
