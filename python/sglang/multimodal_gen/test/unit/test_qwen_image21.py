# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from transformers import BatchFeature

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import (
    QwenImage21ArchConfig,
    QwenImage21DitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import (
    QwenImage21VAEArchConfig,
    QwenImage21VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
    QwenImage21PipelineConfig,
)
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ResidencyState,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
    attend_prefix_segments,
    build_layout,
    pad_prefix_kv,
    prefix_sp_plan,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
    QwenImage21RMS_norm,
    _patchify,
    _unpatchify,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21EncodingStage,
    QwenImage21InputValidationStage,
    collapse_image_slots,
)


@pytest.mark.parametrize("prompt", ["edit", ""])
@pytest.mark.parametrize("image_count", [0, 1, 2])
def test_prompt_conditioning_uses_training_template_and_pre_norm(prompt, image_count):
    hidden = torch.arange(24).reshape(1, 6, 4).float()
    inputs = BatchFeature(
        data={
            "input_ids": torch.tensor([[1, 2, 99, 99, 3, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
        }
    )
    processor = Mock(return_value=inputs)
    processor.tokenizer.convert_tokens_to_ids.return_value = 99
    processor.apply_chat_template.return_value = [[1]]
    encoder = Mock(return_value=SimpleNamespace(hidden_states=(hidden,)))
    stage = QwenImage21EncodingStage(encoder, processor, None, None)
    stage.use_declared_component = Mock(return_value=nullcontext(encoder))
    images = [Image.new("RGBA", (2, 1), (12, 34, 56, 0)) for _ in range(image_count)]
    for image in images:
        image.putpixel((1, 0), (12, 34, 56, 255))
    actual, slots = stage.encode_prompt(prompt, images, "cpu")
    torch.testing.assert_close(actual, hidden[0, [1, 2, 4]])
    assert slots.tolist() == [False, True, False]
    encoder.model.language_model.norm.assert_not_called()
    assert encoder.model.visual.fp32_position_interpolation is False
    kwargs = processor.call_args.kwargs
    prefix = " ".join(
        f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
        for i in range(image_count)
    )
    assert kwargs["text"] == [
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    ]
    assert kwargs["padding_side"] == "left"
    for image in kwargs.get("images", []):
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((1, 0)) == (12, 34, 56)
    for image in images:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0)) == (12, 34, 56, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_encoder_component_offload_preserves_loaded_dtypes(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.managers.memory_managers."
        "component_residency_strategies.get_local_torch_device",
        lambda: torch.device("cuda", torch.cuda.current_device()),
    )
    encoder = torch.nn.Module()
    encoder.model = torch.nn.Module()
    encoder.model.visual = torch.nn.Module()
    encoder.register_parameter(
        "embedding", torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    )
    encoder.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.tensor([0.25, -0.5]).to(torch.float8_e4m3fn), requires_grad=False
        ),
    )
    frequencies = torch.tensor([1.0 / 3, 1.0 / 7])
    encoder.register_buffer("inv_freq", frequencies.clone())
    processor = Mock()
    processor.apply_chat_template.return_value = [[1]]
    stage = QwenImage21EncodingStage(encoder, processor, None, None)
    use = stage.component_uses(None, "conditioning")[0]
    strategy = ComponentOffloadStrategy()
    state = ResidencyState(batch_is_warmup=False)
    weight_bytes = encoder.weight.view(torch.uint8).clone()

    for _ in range(2):
        strategy.prefetch_for_use(encoder, use, state)
        strategy.wait_for_use(encoder, use, state)
        assert encoder.embedding.device.type == "cuda"
        assert encoder.embedding.dtype == torch.bfloat16
        assert encoder.weight.dtype == torch.float8_e4m3fn
        assert encoder.inv_freq.dtype == torch.float32
        torch.testing.assert_close(encoder.inv_freq.cpu(), frequencies, atol=0, rtol=0)
        assert torch.equal(encoder.weight.view(torch.uint8).cpu(), weight_bytes)
        strategy.finish_use(encoder, use, state)
        torch.cuda.synchronize()
        assert encoder.embedding.device.type == "cpu"


def test_condition_slots_expand_to_actual_latent_grid():
    hidden = torch.randn(22, 8)
    ids = torch.tensor([1, 2] + [99] * 16 + [3, 4, 5, 6])
    collapsed, slots = collapse_image_slots(hidden, ids, 99)
    assert collapsed.shape == (7, 8)
    layout = build_layout(slots.tolist(), [(1, 4, 8), (1, 2, 2)], (4, 6, 6), "cpu")
    assert len(layout["image_indices"]) == 32
    assert len(layout["prefix_rope"]) == 38
    assert layout["segments"] == ((0, 2, False), (2, 34, True), (34, 38, False))
    torch.testing.assert_close(collapsed[slots][0], hidden[2])


@pytest.mark.parametrize(
    ("length", "sp_size", "expected"),
    [
        (0, 1, [(0, 0, 0)]),
        (5, 1, [(0, 5, 5)]),
        (5, 2, [(0, 3, 3), (3, 5, 3)]),
        (1, 2, [(0, 1, 1), (1, 1, 1)]),
        (4, 3, [(0, 2, 2), (2, 4, 2), (4, 4, 2)]),
    ],
)
def test_prefix_sp_plan_covers_tokens_without_overlap(length, sp_size, expected):
    plans = [
        prefix_sp_plan(length, sp_size=sp_size, sp_rank=rank) for rank in range(sp_size)
    ]
    assert plans == expected
    covered = []
    for start, end, cap in plans:
        assert 0 <= start <= end <= length
        assert end - start <= cap
        covered.extend(range(start, end))
    assert covered == list(range(length))


def test_prefix_sp_attention_matches_replicated_prefill():
    torch.manual_seed(0)
    length = 7
    segments = ((0, 2, False), (2, 6, True), (6, 7, False))
    q = torch.randn(2, length, 2, 4)
    k = torch.randn(2, length, 2, 4)
    v = torch.randn(2, length, 2, 4)

    def attn(query, key, value, attn_mask=None):
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", weights, value)

    expected = attend_prefix_segments(attn, q, k, v, segments, 0)
    shards, keys, values = [], [], []
    for rank in range(2):
        start, end, cap = prefix_sp_plan(length, sp_size=2, sp_rank=rank)
        keys.append(pad_prefix_kv(k[:, start:end], cap))
        values.append(pad_prefix_kv(v[:, start:end], cap))
        shards.append(
            attend_prefix_segments(attn, q[:, start:end], k, v, segments, start)
        )
    torch.testing.assert_close(torch.cat(keys, dim=1)[:, :length], k, atol=0, rtol=0)
    torch.testing.assert_close(torch.cat(values, dim=1)[:, :length], v, atol=0, rtol=0)
    torch.testing.assert_close(torch.cat(shards, dim=1), expected, atol=0, rtol=0)
    empty = attend_prefix_segments(attn, q[:, :0], k, v, segments, length)
    assert empty.shape == (2, 0, 2, 4)
    with pytest.raises(ValueError, match="local cap"):
        pad_prefix_kv(k, 1)


def test_adjacent_image_slots_stay_distinct():
    layout = build_layout(
        [False, True, True, False], [(1, 2, 2), (1, 4, 2), (1, 2, 2)], (4, 6, 6), "cpu"
    )
    assert layout["segments"] == (
        (0, 1, False),
        (1, 5, True),
        (5, 13, True),
        (13, 14, False),
    )
    with pytest.raises(ValueError, match="slots"):
        build_layout([False], [(1, 2, 2), (1, 2, 2)], (4, 6, 6), "cpu")


def test_latent_pack_decode_contract():
    config = QwenImage21PipelineConfig()
    batch = SimpleNamespace(
        height=64, width=96, extra={"qwen21_positive": {}, "qwen21_negative": {}}
    )
    shape = config.prepare_latent_shape(batch, 2, 1)
    x = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    packed = config.maybe_pack_latents(x, 2, batch)
    assert packed.shape == (2, 24, 64)
    decoded = config.post_denoising_loop(packed, batch)
    assert not batch.extra
    torch.testing.assert_close(decoded[:, :, 0], x[:, 0])
    scale, shift = config.get_decode_scale_and_shift("cpu", torch.float32, None)
    torch.testing.assert_close(
        (decoded.float() - shift) * scale / scale + shift, decoded.float()
    )


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("tiling", [False, True])
def test_native_vae_roundtrip_shapes_and_checkpoint_names(channels, tiling):
    ac = QwenImage21VAEArchConfig(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=(1, 2, 4, 4, 4),
        num_res_blocks=1,
        temperal_downsample=(False, False, False, False),
        in_channels=channels,
        out_channels=channels,
    )
    model = AutoencoderKLQwenImage21(QwenImage21VAEConfig(arch_config=ac)).eval()
    assert not model.use_tiling
    assert ac.scale_factor_spatial == ac.spatial_compression_ratio == 16
    model.use_tiling = tiling
    model.use_parallel_tiling = False
    model.tile_sample_min_height = model.tile_sample_min_width = 32
    model.tile_sample_stride_height = model.tile_sample_stride_width = 16
    with torch.no_grad():
        latent = model.encode(torch.randn(1, channels, 1, 32, 64)).mode()
        assert latent.shape == (1, 4, 1, 2, 4)
        output = model.decode(latent)
        assert output.shape == (1, channels, 1, 32, 64)
    assert model.state_dict()["encoder.conv_in.weight"].ndim == 4
    x = torch.randn(2, 3, 1, 8, 12)
    torch.testing.assert_close(_unpatchify(_patchify(x, 2), 2), x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_vae_rms_norm_normalizes_in_float32(dtype):
    norm = QwenImage21RMS_norm(8, images=False).to(dtype)
    x = torch.linspace(-60000, 60000, 256).reshape(1, 8, 1, 4, 8).to(dtype)
    expected = (
        torch.nn.functional.normalize(x.float(), dim=1).to(dtype)
        * norm.scale
        * norm.gamma
    )
    torch.testing.assert_close(norm(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize("tiling", [False, True])
def test_condition_pixels_match_reference_preprocessing(monkeypatch, tiling):
    module = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21"
    monkeypatch.setattr(f"{module}.get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{module}.set_forward_context", lambda **kwargs: nullcontext())
    image = Image.frombytes("RGBA", (32, 32), bytes(range(256)) * 16)
    vae = Mock()
    vae.encode.return_value.mode.return_value = torch.zeros(1, 64, 1, 2, 2)
    processor = Mock()
    processor.apply_chat_template.return_value = [[1]]
    stage = QwenImage21EncodingStage(Mock(), processor, vae, SimpleNamespace(config={}))
    stage.use_declared_component = Mock(return_value=nullcontext(vae))
    stage.encode_prompt = Mock(
        return_value=(torch.zeros(3, 8), torch.tensor([False, True, False]))
    )
    batch = SimpleNamespace(
        height=32,
        width=32,
        condition_image=image,
        prompt="edit",
        negative_prompt=None,
        num_outputs_per_prompt=1,
        do_classifier_free_guidance=False,
        extra={},
    )
    stage.forward(
        batch,
        SimpleNamespace(pipeline_config=QwenImage21PipelineConfig(vae_tiling=tiling)),
    )
    assert vae.use_tiling is tiling
    expected = VaeImageProcessor(vae_scale_factor=16).preprocess(image).unsqueeze(2)
    actual = vae.encode.call_args.args[0]
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0, rtol=0)
    assert actual.stride() == expected.stride()


def test_condition_image_loading_preserves_alpha(tmp_path):
    path = tmp_path / "condition.png"
    Image.new("RGBA", (32, 32), (12, 34, 56, 78)).save(path)
    image = QwenImage21InputValidationStage().load_condition_image(str(path))
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0)) == (12, 34, 56, 78)
    assert InputValidationStage().load_condition_image(str(path)).mode == "RGB"


def test_architecture_derived_dimensions():
    config = QwenImage21DitConfig(
        arch_config=QwenImage21ArchConfig(num_attention_heads=2, attention_head_dim=16)
    )
    assert config.hidden_size == 32


def test_registry_routes_local_checkpoint_and_preserves_legacy():
    assert (
        _get_config_info("Qwen/Qwen-Image-2.1").pipeline_config_cls
        is QwenImage21PipelineConfig
    )
    assert (
        _get_config_info(
            "/models/private", model_id="Qwen-Image-2.1"
        ).pipeline_config_cls
        is QwenImage21PipelineConfig
    )
    assert (
        _get_config_info("Qwen/Qwen-Image").pipeline_config_cls
        is not QwenImage21PipelineConfig
    )
