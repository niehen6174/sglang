# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ComfyQuant scanner/config and H3 4090 deployment profile."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3ComfyQuant4090PipelineConfig,
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant import (
    ComfyInt8ConvRotLinearMethod,
    ComfyQuantConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant_kernel_adapter import (
    ComfyQuantKernelBackend,
    apply_int8_linear,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant_scanner import (
    ComfyLayerQuantDescription,
    scan_comfy_quant_layers,
)
from sglang.multimodal_gen.runtime.loader import transformer_load_utils as tlu
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _write_comfy_quant_checkpoint(tmp_path: Path) -> str:
    payload = json.dumps(
        {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }
    ).encode("utf-8")
    tensors = {
        "blocks.0.attn.qkv_proj.weight": torch.zeros(4, 8, dtype=torch.int8),
        "blocks.0.attn.qkv_proj.weight_scale": torch.ones(4, 1, dtype=torch.float32),
        "blocks.0.attn.qkv_proj.comfy_quant": torch.tensor(
            list(payload), dtype=torch.uint8
        ),
    }
    path = tmp_path / "dit.safetensors"
    save_file(tensors, str(path))
    return str(path)


def test_scan_comfy_quant_layers_from_tensor_keys(tmp_path):
    ckpt = _write_comfy_quant_checkpoint(tmp_path)
    layers = scan_comfy_quant_layers([ckpt])
    assert set(layers) == {"blocks.0.attn.qkv_proj"}
    desc = layers["blocks.0.attn.qkv_proj"]
    assert desc.format == "int8_tensorwise"
    assert desc.convrot is True
    assert desc.convrot_groupsize == 256


def test_comfy_quant_config_from_safetensors_list(tmp_path):
    ckpt = _write_comfy_quant_checkpoint(tmp_path)
    config = ComfyQuantConfig.from_safetensors_list([ckpt])
    assert config is not None
    assert config.get_name() == "comfy_quant"
    desc = config.layers["blocks.0.attn.qkv_proj"]
    method = ComfyInt8ConvRotLinearMethod(config, desc)
    assert isinstance(method, ComfyInt8ConvRotLinearMethod)


def test_comfy_quant_offload_adapter_disables_dit_cpu_offload():
    args = mock.Mock(spec=ServerArgs)
    args.dit_cpu_offload = True
    adapter = tlu._ComfyQuantOffloadAdapter(
        server_args=args,
        quant_config=ComfyQuantConfig(layers={}),
    )
    adapter.prepare()
    assert args.dit_cpu_offload is False


def test_h3_comfy_4090_deployment_profile():
    config = MiniMaxH3ComfyQuant4090PipelineConfig()
    deployment = config.get_model_deployment_config()
    assert deployment.auto_dit_layerwise_offload is True
    assert deployment.keep_resident_min_available_gb == 20
    assert deployment.keep_resident_components == ("vae",)
    assert config.dit_config.arch_config.use_adaln_curve is True
    assert config.dit_config.arch_config.adaln_input_dim == 8


def test_h3_default_profile_unchanged():
    deployment = MiniMaxH3PipelineConfig().get_model_deployment_config()
    assert deployment.keep_resident_min_available_gb == 120
    assert deployment.keep_resident_components == ("dit", "text_encoder", "vae")


def test_convrot_hadamard_matches_comfy_kitchen():
    from comfy_kitchen.tensor.int8_utils import (
        _build_hadamard as ck_build,
        _rotate_activation as ck_rotate,
    )

    from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant_kernel_adapter import (
        _rotate_convrot_activation,
    )

    torch.manual_seed(0)
    x = torch.randn(2, 512, dtype=torch.bfloat16)
    for groupsize in (16, 64, 256):
        h = ck_build(groupsize, device="cpu", dtype=torch.bfloat16)
        expected = ck_rotate(x, h, groupsize)
        actual = _rotate_convrot_activation(x, groupsize)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_torch_int8_kernel_matches_dequant_reference():
    torch.manual_seed(0)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (32, 16), dtype=torch.int8)
    scale = torch.rand(32, 1, dtype=torch.float32) + 0.1
    bias = torch.randn(32, dtype=torch.float32)

    out = apply_int8_linear(
        backend=ComfyQuantKernelBackend.TORCH,
        x=x,
        weight=weight,
        weight_scale=scale,
        bias=bias,
        convrot=True,
        convrot_groupsize=16,
    )
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


def test_adaln_arch_config_properties():
    arch = MiniMaxH3DiTArchConfig(adaln_curve_grid=1025, adaln_curve_dim=8)
    assert arch.use_adaln_curve is True
    assert arch.adaln_input_dim == 8
    full = MiniMaxH3DiTArchConfig()
    assert full.use_adaln_curve is False
    assert full.adaln_input_dim == full.time_embed_dim


def test_curve_arch_model_skips_time_embedder_on_meta():
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

    config = MiniMaxH3DiTConfig(
        arch_config=MiniMaxH3DiTArchConfig(adaln_curve_grid=1025, adaln_curve_dim=8)
    )
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(config=config, hf_config={})
    assert model.time_embedder is None
    assert model.adaln_t_table.shape == (1025, 8)
    assert "time_embedder.proj_in.weight" not in dict(model.named_parameters())
