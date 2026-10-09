# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.wan import WanAdapter
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.wan import (
    _build_dit_config,
)


@pytest.mark.parametrize("channels,scale", [(16, 8), (48, 16)])
def test_video_geometry_and_odd_spatial_crop(channels, scale):
    x = torch.randn(1, channels, 3, 5, 7)
    adapter = WanAdapter()
    p = adapter.pack(x, torch.tensor([800.0]), torch.randn(1, 12, 4096))
    assert p.latents.shape == (1, channels, 3, 6, 8)
    assert (p.height, p.width, p.num_frames) == (6 * scale, 8 * scale, 9)
    assert p.timesteps.tolist() == [800.0]
    assert torch.equal(adapter.unpack(p.latents, p, x), x)


def test_masked_frame_timesteps_survive_transport():
    x = torch.randn(1, 48, 3, 4, 6)
    t = torch.tensor([[0.0, 800.0, 800.0]])
    p = WanAdapter().pack(x, t, torch.randn(1, 12, 4096))
    assert p.timesteps.shape == (1,)
    assert torch.equal(
        p.extra_req["extra"]["comfyui_model_timestep"], t.repeat_interleave(6, dim=1)
    )
    req = SimpleNamespace(extra=None)
    WanAdapter().fill_req(req, p)
    assert req.extra["comfyui_model_timestep"].shape == (1, 18)


@pytest.mark.parametrize(
    "name", ["clip_fea", "reference_latent", "time_dim_concat", "context_latents"]
)
def test_unimplemented_conditions_fail_explicitly(name):
    with pytest.raises(ValueError, match=name):
        WanAdapter().pack(
            torch.zeros(1, 16, 1, 4, 4),
            torch.ones(1),
            torch.zeros(1, 5, 4096),
            **{name: torch.ones(1)}
        )


@pytest.mark.parametrize("channels", [16, 48])
@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
def test_config_from_checkpoint_header(tmp_path, channels, prefix):
    weights = {
        "patch_embedding.weight": torch.zeros(256, channels, 1, 2, 2),
        "head.head.weight": torch.zeros(channels * 4, 256),
        "time_embedding.0.weight": torch.zeros(256, 128),
        "text_embedding.0.weight": torch.zeros(256, 4096),
        "blocks.0.ffn.0.weight": torch.zeros(512, 256),
        "blocks.1.ffn.0.weight": torch.zeros(512, 256),
    }
    file = tmp_path / "wan.safetensors"
    save_file({prefix + key: value for key, value in weights.items()}, str(file))
    args = SimpleNamespace(model_path=str(file), pipeline_config=WanT2V480PConfig())
    config = _build_dit_config(args)
    assert (config.num_layers, config.num_attention_heads, config.in_channels) == (
        2,
        2,
        channels,
    )
    assert args.pipeline_config.vae_config.arch_config.scale_factor_spatial == (
        16 if channels == 48 else 8
    )
    assert config.ffn_dim == 512


def test_frame_timestep_override_is_not_broadcast_as_schedule():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    t = torch.tensor([[0.0, 0.0, 800.0, 800.0]])
    batch = SimpleNamespace(extra={"comfyui_model_timestep": t})
    result = DenoisingStage.expand_timestep_before_forward(
        None,
        batch,
        SimpleNamespace(comfyui_mode=True),
        torch.tensor(800.0),
        torch.bfloat16,
        None,
        None,
    )
    assert torch.equal(result, t)


def test_schedule_frames_from_adapter():
    p = WanAdapter().pack(
        torch.zeros(1, 16, 5, 4, 4), torch.ones(1), torch.zeros(1, 5, 4096)
    )
    ex = SimpleNamespace(
        adapter=WanAdapter(), model_path="", should_suppress_logs=lambda t: False
    )
    assert (
        SGLDiffusionExecutor._sampling_params_kwargs(ex, p, torch.ones(1))["num_frames"]
        == 17
    )


def test_batch_fallback_preserves_distinct_masked_timesteps():
    class Recorder(SGLDiffusionExecutor):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.adapter = WanAdapter()
            self.sent = []

        def _execute_packed(self, packed, x, timestep):
            self.sent.append(packed)
            return self.adapter.unpack(packed.latents, packed, x)

    ex = Recorder()
    x = torch.randn(2, 48, 3, 4, 4)
    t = torch.tensor([[0.0, 800.0, 800.0], [500.0, 500.0, 500.0]])
    assert torch.equal(ex(x, t, torch.randn(2, 12, 4096)), x)
    assert len(ex.sent) == 2
    assert (
        ex.sent[0].extra_req["extra"]["comfyui_model_timestep"][0, :4].tolist()
        == [0.0] * 4
    )
    assert (
        ex.sent[1].extra_req["extra"]["comfyui_model_timestep"][0, :4].tolist()
        == [500.0] * 4
    )


def test_padding_wraps_like_comfyui():
    x = torch.arange(16 * 1 * 3 * 3).reshape(1, 16, 1, 3, 3).float()
    p = WanAdapter().pack(x, torch.ones(1), torch.zeros(1, 5, 4096))
    assert torch.equal(p.latents[:, :, :, -1, :-1], x[:, :, :, 0, :])
    assert torch.equal(p.latents[:, :, :, :-1, -1], x[:, :, :, :, 0])


def test_frame_timesteps_are_never_restored_from_condition_cache():
    from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
        bind_comfyui_session,
        release_comfyui_session,
    )

    sid = "wan-frame-timestep-test:1"
    first = SimpleNamespace(
        extra={
            "comfyui_session_id": sid,
            "comfyui_cond_key": "same",
            "comfyui_model_timestep": torch.zeros(1, 12),
        },
        prompt_embeds=[torch.ones(1, 5, 32)],
    )
    bind_comfyui_session(first)
    second = SimpleNamespace(
        extra={"comfyui_session_id": sid, "comfyui_cond_key": "same"}, prompt_embeds=[]
    )
    bind_comfyui_session(second)
    assert second.prompt_embeds
    assert "comfyui_model_timestep" not in second.extra
    release_comfyui_session(sid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp32_rope_uses_existing_torch_fallback(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.rotary_embedding import utils

    def unsupported_kernel(**kwargs):
        raise AssertionError("FlashInfer RoPE cannot process FP32")

    monkeypatch.setattr(utils, "flashinfer_apply_rope_inplace", unsupported_kernel)
    q = torch.randn(2, 5, 3, 8, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    angle = torch.randn(5, 4, device="cuda")
    cos = angle.cos()
    sin = angle.sin()
    cache = torch.cat([cos, sin], dim=-1)
    outq, outk = utils.apply_flashinfer_rope_qk_inplace(q, k, cache)

    def reference(x):
        pairs = x.reshape(2, 5, 3, 4, 2)
        rotation = torch.polar(torch.ones_like(angle), angle).reshape(1, 5, 1, 4)
        return torch.view_as_real(torch.view_as_complex(pairs) * rotation).flatten(-2)

    torch.testing.assert_close(outq, reference(q), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(outk, reference(k), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "layer",
    [
        "blocks.0.self_attn.q",
        "blocks.0.cross_attn.o",
        "blocks.0.ffn.0",
        "blocks.0.attn1.to_q",
        "blocks.0.attn2.to_out.0",
        "blocks.0.ffn.net.0.proj",
    ],
)
def test_original_and_diffusers_wan_loras_load_to_runtime_layers(tmp_path, layer):
    from collections import defaultdict

    from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoArchConfig
    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
        get_comfyui_checkpoint_spec,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline

    class Pipeline(LoRAPipeline):
        def create_pipeline_stages(self, server_args):
            pass

    spec = get_comfyui_checkpoint_spec("WanPipeline")
    arch = WanVideoArchConfig()
    assert spec.inherit_config_mapping
    arch.param_names_mapping = {**arch.param_names_mapping, **spec.param_names_mapping}
    file = tmp_path / "adapter.safetensors"
    save_file(
        {
            layer + ".lora_A.weight": torch.ones(2, 4),
            layer + ".lora_B.weight": torch.ones(4, 2),
        },
        str(file),
    )
    pipeline = object.__new__(Pipeline)
    pipeline.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(arch_config=arch)),
        lora_path=None,
        lora_weight_name=None,
    )
    pipeline.modules = {"transformer": torch.nn.Module()}
    pipeline.device = torch.device("cpu")
    pipeline.lora_adapters = defaultdict(dict)
    pipeline.loaded_adapter_paths = {}
    pipeline.loaded_adapter_alphas = {}
    pipeline.load_lora_adapter(str(file), "test", 0)
    target = (
        "blocks.0.attn2.to_out"
        if "cross_attn" in layer or "attn2" in layer
        else ("blocks.0.ffn.fc_in" if "ffn" in layer else "blocks.0.to_q")
    )
    assert set(pipeline.lora_adapters["test"]) == {
        target + ".lora_A",
        target + ".lora_B",
    }
