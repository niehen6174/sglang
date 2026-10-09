# SPDX-License-Identifier: Apache-2.0
"""ComfyUI integrated-mode CFG split: cond and uncond on separate SGLD CFG ranks."""

import uuid
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
    ComfyUIModelAdapter,
    PackedForward,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.cfg_split import (
    cfg_split_unsupported_reason,
)
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    COMFYUI_CFG_SPLIT_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    comfyui_cfg_split as split_stage,
)


class _Adapter(ComfyUIModelAdapter):
    """Minimal adapter (registers no model type): latents as-is, one context."""

    def pack(self, x, timestep, context, **kwargs) -> PackedForward:
        return PackedForward(
            latents=x,
            timesteps=timestep.reshape(-1).float() * 1000.0,
            prompt_embeds=[context.expand(x.shape[0], -1, -1)],
            height=int(x.shape[-2]) * 8,
            width=int(x.shape[-1]) * 8,
        )


class _Executor(SGLDiffusionExecutor):
    adapter_cls = _Adapter
    supports_cfg_split = True


# ----- plugin: which steps split -------------------------------------------------


def _entry(**extra):
    return {"model_conds": {}, "uuid": uuid.uuid4(), **extra}


@pytest.mark.parametrize(
    "conds,options,reason",
    [
        ([[_entry()], [_entry()]], {}, None),
        ([[_entry(strength=0.5)], [_entry(timestep_start=0.9)]], {}, None),
        ([[_entry()]], {}, "1 conds"),
        ([[_entry()], None], {}, "cfg == 1"),
        ([[_entry(), _entry()], [_entry()]], {}, "2 entries"),
        ([[_entry(area=(8, 8, 0, 0))], [_entry()]], {}, "'area'"),
        ([[_entry(mask=torch.ones(1))], [_entry()]], {}, "'mask'"),
        ([[_entry(hooks=object())], [_entry()]], {}, "'hooks'"),
        ([[_entry(control=object())], [_entry()]], {}, "'control'"),
        ([[_entry(gligen=object())], [_entry()]], {}, "'gligen'"),
        ([[_entry(default=True)], [_entry()]], {}, "'default'"),
        ([[_entry()], [_entry()]], {"model_function_wrapper": print}, "model_function"),
        ([[_entry()], [_entry()]], {"multigpu_clones": {}}, "multigpu"),
        (
            [[_entry()], [_entry()]],
            {"transformer_options": {"wrappers": {"apply_model": {"k": [print]}}}},
            "apply_model wrapper",
        ),
        (
            [[_entry()], [_entry()]],
            {"transformer_options": {"wrappers": {"calc_cond_batch": {"k": [print]}}}},
            None,
        ),
    ],
)
def test_split_only_runs_for_plain_cond_uncond_steps(conds, options, reason):
    got = cfg_split_unsupported_reason(conds, options)
    if reason is None:
        assert got is None
    else:
        assert reason in got


@pytest.mark.parametrize(
    "options,ranks",
    [
        ({"num_gpus": 1}, 1),
        ({"num_gpus": 2, "sp_degree": 2}, 1),
        ({"num_gpus": 2, "enable_cfg_parallel": True}, 2),
        ({"num_gpus": 2, "cfg_parallel_degree": 2}, 2),
        ({"num_gpus": 2, "enable_cfg_parallel": True, "tp_size": -1}, 2),
    ],
)
def test_cfg_split_ranks_from_sgld_options(options, ranks):
    assert _Executor.cfg_split_ranks_for(options) == ranks
    # Models without a CFG split stage keep plain enable_cfg_parallel.
    assert SGLDiffusionExecutor.cfg_split_ranks_for(options) == 1


@pytest.mark.parametrize(
    "options,match",
    [
        ({"num_gpus": 1, "enable_cfg_parallel": True}, "num_gpus=2"),
        ({"num_gpus": 4, "enable_cfg_parallel": True}, "num_gpus=2"),
        ({"num_gpus": 2, "enable_cfg_parallel": True, "sp_degree": 2}, "sp_degree=2"),
        ({"num_gpus": 2, "enable_cfg_parallel": True, "tp_size": 2}, "tp_size=2"),
        ({"num_gpus": 2, "cfg_parallel_degree": 4}, "cfg_parallel_degree=4"),
    ],
)
def test_unsupported_cfg_split_combinations_fail_before_the_worker_starts(
    options, match
):
    with pytest.raises(ValueError, match=match):
        _Executor.validate_sgld_options(options)


def _executor():
    config = SimpleNamespace(unet_config={"dtype": torch.bfloat16})
    return _Executor(object(), "m.safetensors", object(), config)


def _packed(executor, context):
    return executor.adapter.pack(torch.zeros(1, 64, 2, 2), torch.tensor([1.0]), context)


def test_conditioning_is_resent_when_a_cond_moves_to_another_rank():
    """A rank only has a cond's conditioning if it was sent to that rank."""
    ex = _executor()
    ctx = torch.randn(1, 3, 8)
    first = _packed(ex, ctx)
    ex._mark_and_maybe_drop(first, slot=0)
    assert first.prompt_embeds  # first time on rank 0: sent
    again = _packed(ex, ctx)
    ex._mark_and_maybe_drop(again, slot=0)
    assert again.prompt_embeds == []  # rank 0 has it
    moved = _packed(ex, ctx)
    ex._mark_and_maybe_drop(moved, slot=1)
    assert moved.prompt_embeds  # rank 1 never saw it
    everyone = _packed(ex, ctx)
    ex._mark_and_maybe_drop(everyone)
    assert everyone.prompt_embeds  # an unsplit call runs on every rank
    later = _packed(ex, ctx)
    ex._mark_and_maybe_drop(later, slot=1)
    assert later.prompt_embeds == []
    ex.begin_sampler_run()
    fresh = _packed(ex, ctx)
    ex._mark_and_maybe_drop(fresh, slot=0)
    assert fresh.prompt_embeds


# ----- plugin: equivalence with ComfyUI's _calc_cond_batch ---------------------------


def _velocity(x, t1000, context):
    """Row-wise stand-in DiT, same arithmetic whichever side calls it."""
    scale = 1 + context.float().mean(dim=(1, 2)).view(-1, 1, 1, 1)
    return (x.float() * scale + t1000.view(-1, 1, 1, 1) / 7).to(x.dtype)


class _Patcher:
    def apply_hooks(self, hooks=None, transformer_options=None):
        return {}

    def prepare_state(self, timestep, model_options):
        pass

    def get_free_memory(self, device):
        return 1 << 40

    def prepare_hook_patches_current_keyframe(self, *a):
        raise AssertionError("no hooks in these tests")


class _Model:
    """The parts of comfy BaseModel that _calc_cond_batch touches."""

    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.current_patcher = _Patcher()

    def memory_required(self, input_shape, cond_shapes=None):
        return 0

    def apply_model(
        self,
        x,
        t,
        c_concat=None,
        c_crossattn=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        out = self.diffusion_model(
            x, t, context=c_crossattn, transformer_options=transformer_options, **kwargs
        )
        return x - out.float() * t.view(-1, 1, 1, 1)


def _reference_dit(x, t, context, **kwargs):
    return _velocity(x, t.reshape(-1).float() * 1000.0, context)


def _worker(requests):
    """Mock SGLD worker: computes each split call and records what it got."""

    rank_state = {}  # what each rank's session keeps per cond

    def send(reqs):
        (req,) = reqs
        requests.append(req)
        calls = req.extra.get(COMFYUI_CFG_SPLIT_KEY)
        if calls is None:
            return SimpleNamespace(
                noise_pred=_velocity(req.latents, req.timesteps, req.prompt_embeds[0]),
                error=None,
            )
        preds = []
        for rank, call in enumerate(calls):
            key = (rank, call["extra"]["comfyui_cond_key"])
            if call["prompt_embeds"]:
                rank_state[key] = call["prompt_embeds"][0]
            preds.append(_velocity(call["latents"], call["timesteps"], rank_state[key]))
        return SimpleNamespace(noise_pred=preds, error=None)

    return send


def _split_executor(monkeypatch, requests):
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors import base

    ex = _executor()
    ex.cfg_split_ranks = 2
    ex.generator = SimpleNamespace(
        server_args=None, _send_to_scheduler_and_wait_for_response=_worker(requests)
    )
    monkeypatch.setattr(
        base.SamplingParams, "from_user_sampling_params_args", lambda *a, **k: None
    )
    monkeypatch.setattr(
        base,
        "prepare_request",
        lambda **k: SimpleNamespace(extra={}, num_outputs_per_prompt=1),
    )
    monkeypatch.setattr(base.torch, "Generator", lambda device: None)
    return ex


def _conds(lengths, strengths=(1.0, 1.0), **extra):
    import comfy.conds

    torch.manual_seed(0)
    return [
        [
            {
                "model_conds": {
                    "c_crossattn": comfy.conds.CONDRegular(torch.randn(1, n, 8))
                },
                "uuid": uuid.uuid4(),
                "strength": s,
                **extra,
            }
        ]
        for n, s in zip(lengths, strengths)
    ]


@pytest.mark.parametrize(
    "lengths,strengths,batch",
    [
        ((5, 9), (1.0, 1.0), 1),  # different lengths: ComfyUI runs two B=1 calls
        ((7, 7), (1.0, 1.0), 1),  # equal lengths: ComfyUI batches them as B=2
        ((5, 9), (0.6, 1.3), 1),  # cond strength
        ((7, 7), (1.0, 1.0), 2),  # latent batch of 2
    ],
)
def test_split_step_equals_comfyui_calc_cond_batch(
    monkeypatch, lengths, strengths, batch
):
    samplers = pytest.importorskip("comfy.samplers")
    conds = _conds(lengths, strengths)
    x = torch.randn(batch, 64, 4, 6, dtype=torch.bfloat16)
    timestep = torch.full((batch,), 0.625)
    options = {"transformer_options": {}}

    expected = samplers._calc_cond_batch(
        _Model(_reference_dit), conds, x, timestep, options
    )

    requests = []
    ex = _split_executor(monkeypatch, requests)
    model = _Model(ex)
    got = ex.calc_cond_batch_wrapper(
        lambda *a: pytest.fail("fell back"), model, conds, x, timestep, options
    )
    assert len(got) == 2
    for g, e in zip(got, expected):
        assert torch.equal(g, e)
    # One worker request per step; cond i rides on CFG rank i.
    (req,) = requests
    calls = req.extra[COMFYUI_CFG_SPLIT_KEY]
    assert [c["prompt_embeds"][0].shape[1] for c in calls] == list(lengths)
    assert all(
        c["extra"]["comfyui_session_id"] == ex.comfyui_session_id() for c in calls
    )


def test_timestep_limited_cond_and_cfg1_use_comfyui_path(monkeypatch):
    pytest.importorskip("comfy.samplers")
    requests = []
    ex = _split_executor(monkeypatch, requests)
    model = _Model(ex)
    x = torch.randn(1, 64, 2, 2)
    timestep = torch.tensor([0.5])
    calls = []

    def comfy_path(*args):
        calls.append(args)
        return ["comfy"]

    # timestep_start below the current sigma: ComfyUI skips that cond.
    limited = _conds((3, 4))
    limited[1][0]["timestep_start"] = 0.25
    assert ex.calc_cond_batch_wrapper(comfy_path, model, limited, x, timestep, {}) == [
        "comfy"
    ]
    one = _conds((3, 4))
    one[1] = None
    assert ex.calc_cond_batch_wrapper(comfy_path, model, one, x, timestep, {}) == [
        "comfy"
    ]
    assert len(calls) == 2 and requests == []
    ex.cfg_split_ranks = 1
    assert ex.calc_cond_batch_wrapper(
        comfy_path, model, _conds((3, 4)), x, timestep, {}
    ) == ["comfy"]


def test_second_step_drops_conditioning_already_on_each_rank(monkeypatch):
    pytest.importorskip("comfy.samplers")
    requests = []
    ex = _split_executor(monkeypatch, requests)
    model = _Model(ex)
    conds = _conds((5, 9))
    x = torch.randn(1, 64, 2, 2, dtype=torch.bfloat16)
    for sigma in (1.0, 0.5):
        ex.calc_cond_batch_wrapper(None, model, conds, x, torch.tensor([sigma]), {})
    first, second = (r.extra[COMFYUI_CFG_SPLIT_KEY] for r in requests)
    assert all(c["prompt_embeds"] for c in first)
    assert all(c["prompt_embeds"] == [] for c in second)
    assert [c["extra"]["comfyui_cond_key"] for c in first] == [
        c["extra"]["comfyui_cond_key"] for c in second
    ]


# ----- worker: routing by CFG rank -----------------------------------------------


class _Group:
    def __init__(self, rank, world_size=2):
        self.rank_in_group, self.world_size = rank, world_size

    def all_gather(self, tensor, dim=0, separate_tensors=False):
        assert separate_tensors
        # Pretend rank r computed tensor + r.
        return [tensor - self.rank_in_group + r for r in range(self.world_size)]


class _Inner:
    def __init__(self):
        self.seen = []

    def forward(self, batch, server_args):
        self.seen.append((batch.latents, dict(batch.extra)))
        batch.noise_pred = batch.latents + 0
        return batch


def _stage(monkeypatch, rank, peers_ok=True):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.base as stage_base

    monkeypatch.setattr(stage_base, "get_global_server_args", lambda: None)
    monkeypatch.setattr(
        split_stage,
        "cfg_rank_errors",
        lambda group, error: [error, None if peers_ok else "KeyError('lost')"],
    )
    inner = _Inner()
    return split_stage.ComfyUICFGSplitStage(
        inner, group_getter=lambda: _Group(rank)
    ), inner


@pytest.mark.parametrize("rank", [0, 1])
def test_each_cfg_rank_runs_its_own_call_and_gathers_all(monkeypatch, rank):
    stage, inner = _stage(monkeypatch, rank)
    calls = [
        {"latents": torch.full((1, 2), 10.0), "extra": {"comfyui_cond_key": "pos"}},
        {"latents": torch.full((1, 2), 20.0), "extra": {"comfyui_cond_key": "neg"}},
    ]
    batch = SimpleNamespace(
        latents=torch.zeros(1, 2),
        extra={"comfyui_session_id": "s:1", COMFYUI_CFG_SPLIT_KEY: calls},
    )
    out = stage.forward(batch, None)
    ((latents, extra),) = inner.seen
    assert torch.equal(latents, calls[rank]["latents"])
    assert extra == {
        "comfyui_session_id": "s:1",
        "comfyui_cond_key": ["pos", "neg"][rank],
    }
    assert len(out.noise_pred) == 2
    assert torch.equal(out.noise_pred[rank], calls[rank]["latents"])


def test_requests_without_split_run_unchanged(monkeypatch):
    stage, inner = _stage(monkeypatch, 1)
    batch = SimpleNamespace(latents=torch.ones(1, 2), extra={"comfyui_session_id": "s"})
    out = stage.forward(batch, None)
    assert torch.equal(out.noise_pred, torch.ones(1, 2))


def test_split_call_count_must_match_cfg_ranks(monkeypatch):
    stage, _ = _stage(monkeypatch, 0)
    batch = SimpleNamespace(latents=None, extra={COMFYUI_CFG_SPLIT_KEY: [{}]})
    with pytest.raises(ValueError, match="1 calls for 2 CFG ranks"):
        stage.forward(batch, None)


def test_list_velocities_gather_per_part():
    """Audio+video models answer [video, audio]; each part gathers on its own."""
    outputs = split_stage.gather_cfg_split_outputs(
        _Group(0), [torch.zeros(1, 2), torch.ones(1, 3)]
    )
    assert len(outputs) == 2
    assert [t.shape for t in outputs[1]] == [(1, 2), (1, 3)]
    assert torch.equal(outputs[1][0], torch.ones(1, 2))


def test_a_failed_cfg_rank_fails_the_step_on_every_rank(monkeypatch):
    """A rank that raised would never reach the all-gather and hang its peer."""
    calls = [{"latents": torch.zeros(1, 2)}, {"latents": torch.ones(1, 2)}]

    stage, inner = _stage(monkeypatch, 0)
    inner.forward = lambda batch, server_args: (_ for _ in ()).throw(KeyError("lost"))
    batch = SimpleNamespace(latents=None, extra={COMFYUI_CFG_SPLIT_KEY: calls})
    with pytest.raises(KeyError, match="lost"):
        stage.forward(batch, None)

    healthy, _ = _stage(monkeypatch, 1, peers_ok=False)
    batch = SimpleNamespace(latents=None, extra={COMFYUI_CFG_SPLIT_KEY: calls})
    with pytest.raises(RuntimeError, match=r"CFG rank 1: KeyError\('lost'\)"):
        healthy.forward(batch, None)


def test_rank_errors_are_exchanged_on_the_cfg_cpu_group(monkeypatch):
    seen = []

    def all_gather_object(out, obj, group=None):
        seen.append(group)
        out[:] = [obj, "boom"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    group = SimpleNamespace(cpu_group="cfg-cpu", world_size=2)
    assert split_stage.cfg_rank_errors(group, None) == [None, "boom"]
    assert seen == ["cfg-cpu"]
