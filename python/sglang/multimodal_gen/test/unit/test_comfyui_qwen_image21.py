# SPDX-License-Identifier: Apache-2.0
"""ComfyUI integrated mode for Qwen-Image 2.1 (``qwen_image21``)."""

from types import SimpleNamespace

import pytest
import torch

try:  # ComfyUI is optional; without a GPU it must start in CPU mode.
    import comfy.cli_args

    if not torch.cuda.is_available():
        comfy.cli_args.args.cpu = True
except ImportError:
    pass

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    PackedForward,
    get_adapter_class,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.qwen_image21 import (
    COND_EXTRA_KEY,
    QwenImage21Adapter,
    QwenImage21Executor,
    qwen_image21_cond_key,
)
from sglang.multimodal_gen.configs.models.dits.qwenimage21 import (
    QwenImage21ArchConfig,
    QwenImage21DitConfig,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import (
    get_comfyui_checkpoint_spec,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.qwen_image21 import (
    arch_from_checkpoint_shapes,
    split_gate_up_markers,
)
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    get_run_state,
    release_comfyui_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    qwen_image21_comfyui as stage_mod,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21_comfyui import (
    QwenImage21ComfyUIStepStage,
    build_comfyui_layout,
    choose_prefix_cache_store,
)

AXES = (16, 56, 56)


def _comfy_shapes(num_layers=32, prefix=""):
    shapes = {
        "img_in.weight": (4096, 64),
        "proj_out.weight": (64, 4096),
        "txt_in.text_norm.weight": (4096,),
        "modulation.1.weight": (16384, 4096),
    }
    for i in range(num_layers):
        shapes[f"transformer_blocks.{i}.attn.norm_q.weight"] = (128,)
        shapes[f"transformer_blocks.{i}.img_mlp.gate_up.weight"] = (24576, 4096)
    return {prefix + k: v for k, v in shapes.items()}


def test_checkpoint_arch_matches_comfyui_detection() -> None:
    arch = arch_from_checkpoint_shapes(_comfy_shapes(prefix="model.diffusion_model."))
    assert arch == dict(
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
    )


def test_checkpoint_spec_splits_fused_gate_up() -> None:
    spec = get_comfyui_checkpoint_spec("QwenImage21Pipeline")
    assert spec.dit_cls_name == "QwenImage21Transformer2DModel"
    config = QwenImage21DitConfig(
        arch_config=QwenImage21ArchConfig(num_attention_heads=1, attention_head_dim=4)
    )
    ffn = 4 * 3
    gate_up = torch.arange(2 * ffn * 4, dtype=torch.float32).reshape(2 * ffn, 4)
    out = dict(
        spec.convert_weights(
            iter(
                [
                    ("transformer_blocks.0.img_mlp.gate_up.weight", gate_up),
                    ("model.diffusion_model.img_in.weight", torch.ones(1)),
                ]
            ),
            config,
        )
    )
    # ComfyUI's swiglu is silu(first half) * second half.
    assert torch.equal(
        out["transformer_blocks.0.img_mlp.gate_layer.weight"], gate_up[:ffn]
    )
    assert torch.equal(out["transformer_blocks.0.img_mlp.proj.weight"], gate_up[ffn:])
    assert "img_in.weight" in out

    with pytest.raises(ValueError):
        list(
            spec.convert_weights(
                iter([("transformer_blocks.0.img_mlp.gate_up.weight", gate_up[:-1])]),
                config,
            )
        )


def test_int8_convrot_gate_up_splits_scale_and_marker() -> None:
    """INT8 ConvRot files carry a per-row weight_scale and a comfy_quant marker per layer."""
    spec = get_comfyui_checkpoint_spec("QwenImage21Pipeline")
    assert spec.quant_markers is not None
    config = QwenImage21DitConfig(
        arch_config=QwenImage21ArchConfig(num_attention_heads=1, attention_head_dim=4)
    )
    ffn = 4 * 3
    scale = torch.arange(2 * ffn, dtype=torch.float32)[:, None]
    out = dict(
        spec.convert_weights(
            iter(
                [
                    ("transformer_blocks.0.img_mlp.gate_up.weight_scale", scale),
                    (
                        "transformer_blocks.0.img_mlp.gate_up.comfy_quant",
                        torch.zeros(3),
                    ),
                ]
            ),
            config,
        )
    )
    assert torch.equal(
        out["transformer_blocks.0.img_mlp.gate_layer.weight_scale"], scale[:ffn]
    )
    assert torch.equal(
        out["transformer_blocks.0.img_mlp.proj.weight_scale"], scale[ffn:]
    )
    assert not any(k.endswith("comfy_quant") for k in out)

    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    markers = split_gate_up_markers(
        {
            "transformer_blocks.0.img_mlp.gate_up": marker,
            "transformer_blocks.0.attn.to_q": marker,
        }
    )
    assert set(markers) == {
        "transformer_blocks.0.img_mlp.gate_layer",
        "transformer_blocks.0.img_mlp.proj",
        "transformer_blocks.0.attn.to_q",
    }


# ----- layout ---------------------------------------------------------------


def _reference_sequence(text_len, slots, ref_shapes, target_hw):
    """ComfyUI ``build_sequence`` token order as (kind, text index, position) plus prefix segments."""
    slots = (list(slots) + [text_len] * len(ref_shapes))[: len(ref_shapes)]
    bounds = [0] + slots + [text_len]
    tokens, segments, pos = [], [], 0
    th, tw = target_hw
    for (start, end), shape in zip(zip(bounds[:-1], bounds[1:]), ref_shapes + [None]):
        if end > start:
            segments.append((len(tokens), len(tokens) + end - start, False))
            tokens += [("text", start + j, (pos + j,) * 3) for j in range(end - start)]
            pos += end - start
        h, w = shape or target_hw
        if shape:
            segments.append((len(tokens), len(tokens) + h * w, True))
        for a in range(h):
            for b in range(w):
                hh = a - (h - h // 2) + 0.5 * (h % 2 - th % 2)
                ww = b - (w - w // 2) + 0.5 * (w % 2 - tw % 2)
                tokens.append(("image" if shape else "target", None, (pos, hh, ww)))
        pos += max(h, w)
    return tokens, segments


@pytest.mark.parametrize(
    "text_len,slots,ref_shapes,target_hw",
    [
        (7, [], [], (4, 6)),
        (7, [3], [(4, 4)], (3, 5)),  # parity differs: half-token reference shift
        (9, [2, 5], [(2, 3), (4, 4)], (4, 4)),
        (6, [], [(2, 2)], (2, 2)),  # slot missing: reference goes after the text
        (
            6,
            [4, 2],
            [(2, 2), (3, 3)],
            (2, 2),
        ),  # unordered slots repeat text like ComfyUI
    ],
)
def test_layout_matches_comfyui_token_order(text_len, slots, ref_shapes, target_hw):
    layout = build_comfyui_layout(
        text_len=text_len,
        image_slots=slots,
        ref_shapes=ref_shapes,
        target_hw=target_hw,
        axes_dims=AXES,
        device="cpu",
    )
    tokens, segments = _reference_sequence(text_len, slots, ref_shapes, target_hw)
    prefix = [t for t in tokens if t[0] != "target"]
    assert layout["encoder_seq_len"] == text_len
    assert layout["prefix_rope"].shape[0] == len(prefix)
    assert layout["target_rope"].shape[0] == target_hw[0] * target_hw[1]
    assert list(layout["segments"]) == segments
    image_rows = set(layout["image_indices"].tolist())
    for i, (kind, src, _) in enumerate(prefix):
        assert (i in image_rows) == (kind == "image")
        if kind == "text":
            assert int(layout["text_indices"][i]) == src
    # The first frequency of each axis rotates by exactly the position.
    rope = torch.cat([layout["prefix_rope"], layout["target_rope"]])
    positions = torch.tensor([t[2] for t in tokens], dtype=torch.float64)
    for axis, col in enumerate([0, AXES[0] // 2, (AXES[0] + AXES[1]) // 2]):
        got = torch.polar(torch.ones_like(positions[:, axis]), positions[:, axis])
        assert torch.allclose(rope[:, col].to(torch.complex128), got, atol=1e-6)


def test_layout_and_rope_equal_comfyui_build_sequence() -> None:
    model_mod = pytest.importorskip("comfy.ldm.qwen_image21.model")
    ops = pytest.importorskip("comfy.ops")
    torch.manual_seed(0)
    model = model_mod.QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=128,
        num_attention_heads=1,
        context_in_dim=8,
        mlp_ratio=1,
        dtype=torch.float32,
        operations=ops.disable_weight_init,
    )
    for p in model.parameters():
        torch.nn.init.normal_(p)
    x = torch.randn(1, 4, 5, 6)
    context = torch.randn(1, 11, 8)
    refs = [torch.randn(1, 4, 4, 4), torch.randn(1, 4, 3, 7)]
    slots = [2, 6]
    with torch.no_grad():
        hidden, pe, segments = model.build_sequence(x, context, refs, slots)
        layout = build_comfyui_layout(
            text_len=11,
            image_slots=slots,
            ref_shapes=[(4, 4), (3, 7)],
            target_hw=(5, 6),
            axes_dims=AXES,
            device="cpu",
        )
        prefix = model.txt_in(context).index_select(1, layout["text_indices"])
        cond = torch.cat([r.flatten(2).transpose(1, 2) for r in refs], dim=1)
        prefix[:, layout["image_indices"]] = model.img_in(cond)
    prefix_len = layout["prefix_rope"].shape[0]
    # Same per-token projections; only the GEMM batching differs.
    torch.testing.assert_close(prefix, hidden[:, :prefix_len], rtol=1e-5, atol=1e-4)
    rope = torch.cat([layout["prefix_rope"], layout["target_rope"]])
    # ComfyUI pe: (1, N, 1, D/2, 2, 2) rotation matrices [[cos, -sin], [sin, cos]].
    assert torch.equal(rope.real, pe[0, :, 0, :, 0, 0])
    assert torch.equal(rope.imag, pe[0, :, 0, :, 1, 0])
    comfy_prefix = [(s, e, m is None) for s, e, m in segments[:-1]]
    assert list(layout["segments"]) == comfy_prefix
    assert segments[-1][:2] == (prefix_len, hidden.shape[1])


# ----- adapter / executor ------------------------------------------------------


def test_qwen_image21_adapter_is_registered() -> None:
    assert get_adapter_class("qwen_image21") is QwenImage21Adapter
    assert QwenImage21Adapter.pipeline_class_name == "QwenImage21Pipeline"
    assert stage_mod.COND_EXTRA_KEY == COND_EXTRA_KEY


def test_pack_keeps_comfy_latents_and_sends_payload() -> None:
    adapter = QwenImage21Adapter()
    x = torch.randn(2, 64, 5, 7)
    ref = torch.randn(2, 64, 4, 4)
    sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0])
    packed = adapter.pack(
        x,
        torch.tensor([0.5, 0.5]),
        torch.randn(1, 9, 32),
        ref_latents=[ref],
        image_slots=[3],
        transformer_options={
            "sample_sigmas": sigmas,
            "qwen_image21_cache": {"device": "cpu", "dtype": "int8"},
        },
    )
    assert packed.latents is x
    assert torch.equal(packed.timesteps, torch.tensor([500.0, 500.0]))
    assert packed.prompt_embeds[0].shape == (2, 9, 32)
    assert (packed.height, packed.width) == (80, 112)
    payload = packed.extra_req[COND_EXTRA_KEY]
    assert payload["image_slots"] == [3]
    assert payload["ref_latents"][0] is ref
    assert payload["prefix_cache"] == "cpu"
    assert payload["last_sigma"] == 0.25

    req = SimpleNamespace(extra={"comfyui_session_id": "s:1"})
    adapter.fill_req(req, packed)
    assert req.extra[COND_EXTRA_KEY] is payload
    assert req.extra["comfyui_session_id"] == "s:1"

    adapter.drop_cached_fields(packed)
    assert packed.prompt_embeds == []
    assert COND_EXTRA_KEY not in packed.extra_req

    pred = torch.randn(2, 64, 5, 7)
    assert adapter.unpack(pred, packed, x) is pred


def test_pack_rejects_dit_patches() -> None:
    with pytest.raises(NotImplementedError):
        QwenImage21Adapter().pack(
            torch.randn(1, 64, 2, 2),
            torch.tensor([1.0]),
            torch.randn(1, 3, 8),
            transformer_options={"patches_replace": {"dit": {("single_block", 0): 1}}},
        )


def test_cond_key_separates_prompts_sharing_head_and_tail() -> None:
    def packed(context, slots=(), refs=()):
        return PackedForward(
            latents=torch.zeros(1),
            timesteps=torch.zeros(1),
            prompt_embeds=[context],
            height=16,
            width=16,
            extra_req={
                COND_EXTRA_KEY: {"image_slots": list(slots), "ref_latents": list(refs)}
            },
        )

    pos = torch.randn(1, 6, 8)
    neg = pos.clone()
    neg[0, 2:4] += 1.0  # same first and last element, same shape
    assert qwen_image21_cond_key(packed(pos)) == qwen_image21_cond_key(
        packed(pos.clone())
    )
    assert qwen_image21_cond_key(packed(pos)) != qwen_image21_cond_key(packed(neg))
    assert qwen_image21_cond_key(packed(pos, [2])) != qwen_image21_cond_key(packed(pos))
    ref = torch.randn(1, 4, 2, 2)
    assert qwen_image21_cond_key(packed(pos, [2], [ref])) != qwen_image21_cond_key(
        packed(pos, [2], [ref + 1])
    )


def test_executor_accepts_comfyui_prefix_cache_hooks() -> None:
    config = SimpleNamespace(unet_config={"dtype": torch.bfloat16})
    executor = QwenImage21Executor(object(), "m.safetensors", object(), config)
    # ComfyUI QwenImage21.current_patcher setter calls both on pre_run / cleanup.
    executor.current_patcher = object()
    executor.reset_prefix_cache(True)
    executor.reset_prefix_cache(False)
    assert executor.should_suppress_logs(torch.tensor([0.5, 0.5]))


# ----- worker stage ------------------------------------------------------------


@pytest.mark.parametrize(
    "option,free_gpu,free_host,expected",
    [
        ("off", 100, 100, None),
        ("auto", 41, 0, "gpu"),
        ("auto", 40, 100, "cpu"),
        ("auto", 40, 10, None),
        ("gpu", 21, 0, "gpu"),
        ("gpu", 20, 100, None),
        ("cpu", 100, 21, "cpu"),
    ],
)
def test_prefix_cache_store_follows_comfyui_rules(
    option, free_gpu, free_host, expected
):
    assert (
        choose_prefix_cache_store(
            option, need_bytes=10, free_gpu_bytes=free_gpu, free_host_bytes=free_host
        )
        == expected
    )


class _FakeDiT:
    num_layers = 2

    def __init__(self):
        self.calls = []

    def __call__(self, *, hidden_states, timestep, prefix_caches, layouts, **kwargs):
        prefilled = []
        for caches in prefix_caches or []:
            prefilled.append(bool(caches[0]))
            for layer in caches:
                if not layer:
                    layer.update(key=torch.ones(1), value=torch.ones(1))
        self.calls.append(
            dict(
                prefix_caches=prefix_caches,
                prefilled=prefilled,
                layouts=layouts,
                timestep=timestep,
                **kwargs,
            )
        )
        return hidden_states * 2


def _stage(monkeypatch, free_gpu=1 << 40):
    stage = QwenImage21ComfyUIStepStage.__new__(QwenImage21ComfyUIStepStage)
    arch = QwenImage21ArchConfig(
        num_layers=2, num_attention_heads=1, attention_head_dim=128
    )
    stage.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=QwenImage21DitConfig(arch_config=arch)
        )
    )
    fake = _FakeDiT()
    stage.transformer = fake
    stage._predict_noise = lambda current_model, latent_model_input, **kw: (
        current_model(hidden_states=latent_model_input, **kw)
    )
    monkeypatch.setattr(stage_mod, "_free_memory", lambda device: (free_gpu, 1 << 40))
    monkeypatch.setattr(stage_mod, "get_sp_world_size", lambda: 1)
    return stage, fake


def _req(sid, cond_key, latents, sigma, context=None, payload=None):
    extra = {"comfyui_session_id": sid, "comfyui_cond_key": cond_key}
    if payload is not None:
        extra[COND_EXTRA_KEY] = payload
    return SimpleNamespace(
        extra=extra,
        latents=latents,
        timesteps=torch.full((latents.shape[0],), sigma * 1000.0),
        prompt_embeds=[] if context is None else [context],
    )


def test_step_stage_keeps_prefix_cache_per_cond(monkeypatch) -> None:
    stage, fake = _stage(monkeypatch)
    sid = "exec:1"
    x = torch.randn(1, 64, 3, 4)
    pos, neg = torch.randn(1, 5, 16), torch.randn(1, 7, 16)
    payload = {
        "image_slots": [],
        "ref_latents": [],
        "prefix_cache": "auto",
        "last_sigma": 0.5,
    }
    try:
        for sigma in (1.0, 0.5):
            first = sigma == 1.0
            for key, ctx in (("pos", pos), ("neg", neg)):
                req = _req(
                    sid,
                    key,
                    x,
                    sigma,
                    context=ctx if first else None,
                    payload=dict(payload) if first else None,
                )
                out = stage.forward(req, None)
                assert torch.equal(out.noise_pred, x.to(torch.bfloat16) * 2)
        state = get_run_state(SimpleNamespace(extra={"comfyui_session_id": sid}))
        assert len(state.conds) == 2
        pos_calls = fake.calls[0::2]
        assert pos_calls[0]["prefilled"] == [False]
        assert pos_calls[1]["prefilled"] == [True]
        assert torch.equal(
            pos_calls[1]["encoder_hidden_states"], pos.to(torch.bfloat16)
        )
        assert fake.calls[1]["encoder_hidden_states"].shape[1] == 7
        # last_sigma reached: the K/V is released, conditioning kept for a late call.
        assert all(c.prefix_caches is None for c in state.conds.values())
        stage.forward(_req(sid, "pos", x, 0.5), None)
        assert fake.calls[-1]["prefix_caches"] is None

        # A new sampler run of the same executor evicts the old run.
        stage.forward(_req("exec:2", "pos", x, 1.0, context=pos, payload=payload), None)
        assert get_run_state(SimpleNamespace(extra={"comfyui_session_id": sid})) is None
    finally:
        release_comfyui_session(sid)
        release_comfyui_session("exec:2")


def test_step_stage_batched_cond_uses_one_cache_per_row(monkeypatch) -> None:
    stage, fake = _stage(monkeypatch)
    x = torch.randn(2, 64, 2, 2)
    ref = torch.randn(1, 64, 2, 2)
    payload = {"image_slots": [1], "ref_latents": [ref], "prefix_cache": "gpu"}
    try:
        stage.forward(
            _req("b:1", "k", x, 1.0, context=torch.randn(2, 3, 16), payload=payload),
            None,
        )
        call = fake.calls[-1]
        assert len(call["prefix_caches"]) == 2
        assert call["prefix_caches"][0] is not call["prefix_caches"][1]
        assert len(call["layouts"]) == 2
        assert call["condition_latents"].shape == (2, 4, 64)
        assert call["timestep"].shape == (2,)
    finally:
        release_comfyui_session("b:1")


def test_step_stage_recomputes_when_cache_is_off(monkeypatch) -> None:
    stage, fake = _stage(monkeypatch)
    payload = {"image_slots": [], "ref_latents": [], "prefix_cache": "off"}
    try:
        x = torch.randn(1, 64, 2, 2)
        stage.forward(
            _req("o:1", "k", x, 1.0, context=torch.randn(1, 3, 16), payload=payload),
            None,
        )
        stage.forward(_req("o:1", "k", x, 0.5), None)
        assert [c["prefix_caches"] for c in fake.calls] == [None, None]
        assert fake.calls[1]["encoder_hidden_states"].shape == (1, 3, 16)
    finally:
        release_comfyui_session("o:1")


def test_step_stage_without_conditioning_fails_loudly(monkeypatch) -> None:
    stage, _ = _stage(monkeypatch)
    with pytest.raises(RuntimeError, match="no conditioning"):
        stage.forward(_req("lost:1", "k", torch.randn(1, 64, 2, 2), 0.5), None)
    release_comfyui_session("lost:1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_host_prefix_kv_roundtrip() -> None:
    caches = stage_mod.make_prefix_caches(
        "cpu", batch=1, num_layers=3, device=torch.device("cuda")
    )
    kv = [
        (torch.randn(1, 5, 2, 4, device="cuda"), torch.randn(1, 5, 2, 4, device="cuda"))
        for _ in range(3)
    ]
    for slot, (k, v) in zip(caches[0], kv):
        assert not slot
        slot.update(key=k, value=v)
        assert slot
    for _ in range(2):
        for slot, (k, v) in zip(caches[0], kv):
            assert slot["key"].is_cuda
            assert torch.equal(slot["key"], k)
            assert torch.equal(slot["value"], v)


def test_pipeline_installs_comfyui_step_stage(monkeypatch) -> None:
    from sglang.multimodal_gen.runtime.pipelines.qwen_image21 import QwenImage21Pipeline

    created = {}

    def _fake_init(self, transformer, scheduler):
        created["modules"] = (transformer, scheduler)

    monkeypatch.setattr(QwenImage21ComfyUIStepStage, "__init__", _fake_init)

    class _Pipe:
        def __init__(self):
            self.modules = {"transformer": object(), "scheduler": object()}
            self.stages = []

        def get_module(self, name):
            return self.modules[name]

        def add_stage(self, stage, stage_name=None):
            self.stages.append(stage)

    pipe = _Pipe()
    QwenImage21Pipeline.create_comfyui_stages(
        pipe, server_args=SimpleNamespace(enable_cfg_parallel=False)
    )
    assert [type(s) for s in pipe.stages] == [QwenImage21ComfyUIStepStage]
    assert created["modules"] == (
        pipe.modules["transformer"],
        pipe.modules["scheduler"],
    )
    assert (
        QwenImage21Pipeline.pipeline_config_cls.__name__ == "QwenImage21PipelineConfig"
    )
    import sglang.multimodal_gen.runtime.pipelines_core.stages.base as stage_base
    from sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_cfg_split import (
        ComfyUICFGSplitStage,
    )

    monkeypatch.setattr(stage_base, "get_global_server_args", lambda: None)
    split = _Pipe()
    QwenImage21Pipeline.create_comfyui_stages(
        split,
        server_args=SimpleNamespace(enable_cfg_parallel=True, sp_degree=1, tp_size=1),
    )
    # CFG split: the step stage runs one cond per CFG rank.
    (wrapper,) = split.stages
    assert isinstance(wrapper, ComfyUICFGSplitStage)
    assert isinstance(wrapper.step_stage, QwenImage21ComfyUIStepStage)
    for parallel in ({"sp_degree": 2, "tp_size": 1}, {"sp_degree": 1, "tp_size": 2}):
        with pytest.raises(ValueError, match="not supported"):
            QwenImage21Pipeline.create_comfyui_stages(
                _Pipe(),
                server_args=SimpleNamespace(enable_cfg_parallel=True, **parallel),
            )


# ----- multi-GPU -----------------------------------------------------------------


def test_qwen_image21_runs_cfg_split() -> None:
    options = {"num_gpus": 2, "enable_cfg_parallel": True}
    assert QwenImage21Executor.cfg_split_ranks_for(options) == 2


def test_sp_needs_a_token_count_divisible_by_the_sp_degree(monkeypatch) -> None:
    """An odd latent grid under SP used to fail deep in the DiT as a None noise_pred."""
    stage, fake = _stage(monkeypatch)
    monkeypatch.setattr(stage_mod, "get_sp_world_size", lambda: 2)
    payload = {"image_slots": [], "ref_latents": [], "prefix_cache": "auto"}
    try:
        with pytest.raises(ValueError, match="3969 tokens"):
            stage.forward(
                _req(
                    "sp:1",
                    "k",
                    torch.randn(1, 64, 63, 63),
                    1.0,
                    torch.randn(1, 3, 16),
                    payload,
                ),
                None,
            )
        stage.forward(
            _req(
                "sp:1",
                "k",
                torch.randn(1, 64, 63, 64),
                1.0,
                torch.randn(1, 3, 16),
                payload,
            ),
            None,
        )
        assert fake.calls
    finally:
        release_comfyui_session("sp:1")


def test_prefix_cache_placement_is_agreed_across_ranks(monkeypatch) -> None:
    """Under TP an uncached prefix runs extra all-reduces, so ranks must not split.

    Only ranks that run the same DiT call agree; CFG-split ranks run different
    conds at different times, so including them would deadlock.
    """
    seen = []

    def all_reduce(tensor, op=None, group=None):
        seen.append(group)
        tensor.copy_(torch.minimum(tensor, torch.tensor([3, 5])))

    def group(name, size):
        return SimpleNamespace(world_size=size, cpu_group=name)

    monkeypatch.setattr(stage_mod, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(stage_mod, "get_tp_group", lambda: group("tp", 2))
    monkeypatch.setattr(stage_mod, "get_sp_group", lambda: group("sp", 1))
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    free_gpu, free_host = stage_mod._free_memory(torch.device("cpu"))
    assert (free_gpu, free_host) == (0, 5)
    assert seen == ["tp"]

    # CFG split (tp = sp = 1): no collective at all.
    seen.clear()
    monkeypatch.setattr(stage_mod, "get_tp_group", lambda: group("tp", 1))
    stage_mod._free_memory(torch.device("cpu"))
    assert seen == []


def test_failed_worker_step_raises_with_the_worker_error(monkeypatch) -> None:
    """A worker-side exception returns noise_pred=None; it must not become AttributeError."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors import base

    config = SimpleNamespace(unet_config={"dtype": torch.bfloat16})
    generator = SimpleNamespace(
        server_args=None,
        _send_to_scheduler_and_wait_for_response=lambda reqs: SimpleNamespace(
            noise_pred=None, error="boom on rank 0"
        ),
    )
    executor = QwenImage21Executor(generator, "m.safetensors", object(), config)
    monkeypatch.setattr(
        base.SamplingParams, "from_user_sampling_params_args", lambda *a, **k: None
    )
    monkeypatch.setattr(
        base,
        "prepare_request",
        lambda **k: SimpleNamespace(extra={}, num_outputs_per_prompt=1),
    )
    monkeypatch.setattr(base.torch, "Generator", lambda device: None)
    x = torch.randn(1, 64, 2, 2)
    packed = executor.adapter.pack(x, torch.tensor([1.0]), torch.randn(1, 3, 8))
    with pytest.raises(RuntimeError, match="boom on rank 0"):
        executor._execute_packed(packed, x, torch.tensor([1.0]))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="ServerArgs checks num_gpus")
def test_comfyui_mode_never_auto_enables_cfg_parallel(tmp_path, monkeypatch) -> None:
    """num_gpus=2 alone looked up model_index.json for a single-file DiT and failed."""
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    def must_not_run(self):
        raise AssertionError("default-CFG lookup in comfyui_mode")

    monkeypatch.setattr(ServerArgs, "_model_default_uses_cfg", must_not_run)
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"")
    args = ServerArgs.from_kwargs(
        model_path=str(dit),
        pipeline_class_name="QwenImage21Pipeline",
        comfyui_mode=True,
        num_gpus=2,
    )
    assert not args.enable_cfg_parallel
    assert args.sp_degree == 2
