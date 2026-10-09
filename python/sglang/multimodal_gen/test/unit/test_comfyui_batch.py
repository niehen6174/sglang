# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.zimage import (
    ZImageAdapter,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


def _adapter_class(name):
    if name == "flux":
        return FluxAdapter
    module = pytest.importorskip(
        "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.qwen_image"
    )
    return (
        module.QwenImageEditAdapter if name == "qwen_edit" else module.QwenImageAdapter
    )


class RecordingExecutor(SGLDiffusionExecutor):
    def __init__(self, adapter):
        torch.nn.Module.__init__(self)
        self.adapter = adapter
        self.sent = []
        self.generator = SimpleNamespace(
            server_args=SimpleNamespace(comfyui_native_batch=True)
        )

    def _execute_packed(self, packed, x, timestep):
        self.sent.append((packed, timestep))
        return self.adapter.unpack(packed.latents, packed, x)


@pytest.mark.parametrize("adapter", ["flux", "qwen"])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_native_batch_keeps_rows_and_sends_one_step(adapter, batch):
    adapter = _adapter_class(adapter)
    ex = RecordingExecutor(adapter())
    shape = (batch, 16, 8, 8) if adapter is FluxAdapter else (batch, 16, 1, 8, 8)
    x = torch.randn(shape)
    context = torch.randn(batch, 7, 32)
    y = torch.randn(batch, 768)
    t = torch.full((batch,), 0.5)
    out = ex(x, t, context, y=y)
    assert len(ex.sent) == 1
    packed, _ = ex.sent[0]
    assert packed.latents.shape[0] == batch
    assert packed.timesteps.tolist() == [500.0]
    assert packed.prompt_seq_lens[-1] == [7] * batch
    assert torch.equal(out, x)
    assert torch.equal(packed.prompt_embeds[-1], context)
    if adapter is FluxAdapter:
        assert torch.equal(packed.pooled_embeds[0], y)


@pytest.mark.parametrize("adapter", ["flux", "qwen"])
def test_different_timesteps_are_not_collapsed(adapter):
    adapter = _adapter_class(adapter)
    ex = RecordingExecutor(adapter())
    x = torch.randn((2, 16, 8, 8) if adapter is FluxAdapter else (2, 16, 1, 8, 8))
    out = ex(x, torch.tensor([0.5, 0.7]), torch.randn(2, 7, 32), y=torch.randn(2, 768))
    assert len(ex.sent) == 2
    assert [p.timesteps.item() for p, _ in ex.sent] == [500.0, 700.0]
    assert torch.equal(out, x)


def test_different_flux_guidance_is_preserved_per_row():
    ex = RecordingExecutor(FluxAdapter())
    x = torch.randn(2, 16, 8, 8)
    ex(
        x,
        torch.full((2,), 0.5),
        torch.randn(2, 7, 32),
        y=torch.randn(2, 768),
        guidance=torch.tensor([2.0, 4.0]),
    )
    assert [p.guidance_scale for p, _ in ex.sent] == [2.0, 4.0]


def test_qwen_edit_references_remain_per_row():
    ex = RecordingExecutor(_adapter_class("qwen_edit")())
    x, refs = torch.randn(2, 16, 1, 8, 8), torch.randn(2, 16, 1, 8, 8)
    out = ex(x, torch.full((2,), 0.5), torch.randn(2, 7, 32), ref_latents=[refs])
    assert len(ex.sent) == 2
    for i, (packed, _) in enumerate(ex.sent):
        expected, _ = ex.adapter._pack_latents(refs[i : i + 1])
        assert torch.equal(packed.extra_req["image_latent"], expected)
    assert torch.equal(out, x)


def test_zimage_keeps_per_sample_text_contract():
    ex = RecordingExecutor(ZImageAdapter())
    x = torch.randn(2, 16, 8, 8)
    out = ex(x, torch.full((2,), 0.5), torch.randn(2, 7, 32))
    assert len(ex.sent) == 2
    assert all(p.prompt_embeds[0].shape == (7, 32) for p, _ in ex.sent)
    assert torch.equal(out, x)


def test_batched_request_has_two_generators_even_after_condition_cache_hit():
    ex = RecordingExecutor(FluxAdapter())
    ex.model_path = "/test-model"
    ex.session_id, ex._run_id, ex._sent_conds = "batch-test", 0, set()

    def send(requests):
        req = requests[0]
        assert req.batch_size == 2
        assert len(req.generator) == 2
        assert req.timesteps.numel() == 1
        return SimpleNamespace(noise_pred=req.latents)

    ex.generator = SimpleNamespace(
        server_args=SimpleNamespace(attention_backend_config={}, enable_trace=False),
        _send_to_scheduler_and_wait_for_response=send,
    )
    x, t, context, y = (
        torch.randn(2, 16, 8, 8),
        torch.full((2,), 0.5),
        torch.randn(2, 7, 32),
        torch.randn(2, 768),
    )

    with patch(
        "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base.SamplingParams.from_user_sampling_params_args",
        side_effect=lambda model_path, server_args, **kw: SamplingParams(**kw),
    ), patch(
        "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base.torch.Generator",
        side_effect=lambda device: object(),
    ):
        for _ in range(2):
            packed = ex.adapter.pack(x, t, context, y=y)
            packed.timesteps = packed.timesteps[:1]
            out = SGLDiffusionExecutor._execute_packed(ex, packed, x, t)
            assert torch.equal(out, x)
        assert packed.prompt_embeds == []


@pytest.mark.parametrize("guided", [False, True])
def test_flux_checkpoint_detects_optional_guidance_weights(tmp_path, guided):
    from safetensors.torch import save_file
    from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.flux import (
        _build_dit_config,
    )

    path = tmp_path / "flux.safetensors"
    weights = {"img_in.weight": torch.zeros(2, 2)}
    if guided:
        weights["guidance_in.in_layer.weight"] = torch.zeros(2, 2)
    save_file(weights, str(path))
    args = SimpleNamespace(
        model_path=str(path), pipeline_config=SimpleNamespace(dit_config=FluxConfig())
    )
    assert _build_dit_config(args).arch_config.guidance_embeds is guided


@pytest.mark.parametrize("enabled", [True, False])
def test_options_preserves_bf16_reduction_setting(enabled):
    import ast
    from pathlib import Path

    path = Path(__file__).parents[2] / "apps/ComfyUI_SGLDiffusion/nodes.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SGLDOptions"
    )
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    options = namespace["SGLDOptions"]().create_options(
        allow_bf16_reduced_precision_reduction=enabled
    )[0]
    assert options["allow_bf16_reduced_precision_reduction"] is enabled


@pytest.mark.parametrize("adapter", ["flux", "qwen"])
def test_default_mode_uses_per_row_reference(adapter):
    adapter = _adapter_class(adapter)
    ex = RecordingExecutor(adapter())
    ex.generator.server_args.comfyui_native_batch = False
    x = torch.randn((2, 16, 8, 8) if adapter is FluxAdapter else (2, 16, 1, 8, 8))
    out = ex(x, torch.full((2,), 0.5), torch.randn(2, 7, 32), y=torch.randn(2, 768))
    assert len(ex.sent) == 2
    assert torch.equal(out, x)


@pytest.mark.parametrize("enabled", [True, False])
def test_options_preserves_native_batch_setting(enabled):
    import ast
    from pathlib import Path

    path = Path(__file__).parents[2] / "apps/ComfyUI_SGLDiffusion/nodes.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SGLDOptions"
    )
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    options = namespace["SGLDOptions"]().create_options(enable_native_batch=enabled)[0]
    assert options["comfyui_native_batch"] is enabled
