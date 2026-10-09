# SPDX-License-Identifier: Apache-2.0
"""Process-wide SGLD worker ownership for ComfyUI loaders."""

from types import SimpleNamespace

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
    SGLDiffusionGenerator,
)


def test_shared_is_process_singleton() -> None:
    SGLDiffusionGenerator.reset_shared()
    assert SGLDiffusionGenerator.shared() is SGLDiffusionGenerator.shared()
    SGLDiffusionGenerator.reset_shared()


def test_reuse_requires_live_worker() -> None:
    runtime = SGLDiffusionGenerator()
    options = {"model_path": "z.safetensors"}
    runtime.last_options = options
    runtime.generator = object()
    runtime._patcher = object()
    runtime.executor = object()
    runtime._is_live = lambda: False
    assert runtime._can_reuse(options) is False
    runtime._is_live = lambda: True
    assert runtime._can_reuse(options) is True
    assert runtime._can_reuse({"model_path": "h3.safetensors"}) is False


def test_ensure_rebuilds_when_another_model_owns_the_worker() -> None:
    runtime = SGLDiffusionGenerator()
    stale = object()
    fresh = object()
    loads = []
    executor = SimpleNamespace(
        generator=stale,
        _sgld_reload={
            "model_path": "z.safetensors",
            "model_options": {},
            "sgld_options": {},
        },
        _lora_input=None,
    )

    def fake_load(**kwargs):
        loads.append(kwargs)
        runtime.generator = fresh
        return "patcher"

    runtime.load_model = fake_load
    runtime.ensure_executor(executor)
    assert loads == [executor._sgld_reload]
    assert executor.generator is fresh


def test_ensure_is_noop_when_executor_still_owns_live_worker() -> None:
    runtime = SGLDiffusionGenerator()
    gen = object()
    runtime.generator = gen
    runtime._is_live = lambda: True
    executor = SimpleNamespace(generator=gen, _sgld_reload={"model_path": "z"})
    runtime.load_model = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("should not reload")
    )
    runtime.ensure_executor(executor)
    assert executor.generator is gen


def test_kill_generator_only_touches_owned_workers() -> None:
    runtime = SGLDiffusionGenerator()
    owned = SimpleNamespace(alive=True, terminated=False, killed=False, pid=9)

    def terminate():
        owned.terminated = True
        owned.alive = False

    owned.is_alive = lambda: owned.alive
    owned.terminate = terminate
    owned.join = lambda timeout=None: None
    owned.kill = lambda: setattr(owned, "killed", True)
    runtime.generator = SimpleNamespace(local_scheduler_process=[owned])
    runtime.kill_generator()
    assert owned.terminated is True
    assert owned.killed is False


def test_sampler_selects_lora_from_each_patcher_and_clears_previous_adapter():
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    events = []
    payload = {"lora_path": "four-step.safetensors", "strength": 1.0}
    runtime = SimpleNamespace(
        _lora_input=payload.copy(),
        generator=SimpleNamespace(unmerge_lora_weights=lambda: events.append("clear")),
        begin_sampler_run=lambda: events.append("begin"),
        end_sampler_run=lambda: events.append("end"),
    )

    def set_lora(**desired):
        events.append(("load", desired))
        runtime._lora_input = desired.copy()

    runtime.set_lora = set_lora
    base = SimpleNamespace(model_patcher=SimpleNamespace(model_options={}))
    adapted = SimpleNamespace(
        model_patcher=SimpleNamespace(model_options={"sgld_lora_input": payload})
    )
    sample = lambda *args, **kwargs: "sampled"
    assert (
        SGLDiffusionExecutor.sampler_sample_wrapper(runtime, sample, base) == "sampled"
    )
    assert runtime._lora_input is None
    assert events == ["clear", "begin", "end"]
    events.clear()
    SGLDiffusionExecutor.sampler_sample_wrapper(runtime, sample, adapted)
    SGLDiffusionExecutor.sampler_sample_wrapper(runtime, sample, adapted)
    assert events.count(("load", payload)) == 1
    assert "clear" not in events


def test_cached_base_model_resets_request_accelerations_after_spectrum_run():
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    runtime = SimpleNamespace(
        _lora_input=None, begin_sampler_run=lambda: None, end_sampler_run=lambda: None
    )
    accelerated = SimpleNamespace(
        model_patcher=SimpleNamespace(
            model_options={
                "sgld_request_flags": {
                    "enable_cache_dit": True,
                    "cache_dit_params": {"residual_diff_threshold": 0.12},
                    "request_options": {"enable_spectrum": True},
                }
            }
        )
    )
    base = SimpleNamespace(model_patcher=SimpleNamespace(model_options={}))
    SGLDiffusionExecutor.sampler_sample_wrapper(
        runtime, lambda *args: None, accelerated
    )
    assert runtime.request_options == {"enable_spectrum": True}
    assert runtime.enable_cache_dit is True
    SGLDiffusionExecutor.sampler_sample_wrapper(runtime, lambda *args: None, base)
    assert runtime.request_options == {}
    assert runtime.enable_cache_dit is None
    assert runtime.cache_dit_params is None


def test_failed_lora_request_does_not_claim_adapter_is_active():
    import pytest
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
        SGLDiffusionExecutor,
    )

    def reject(**kwargs):
        raise RuntimeError("Dynamic LoRA supports only one adapter")

    runtime = SimpleNamespace(
        _lora_input=None, generator=SimpleNamespace(set_lora=reject)
    )
    with pytest.raises(RuntimeError, match="one adapter"):
        SGLDiffusionExecutor.set_lora(
            runtime,
            lora_nickname=["a", "b"],
            lora_path=["a.safetensors", "b.safetensors"],
            strength=[1, 0.25],
            target=["all", "all"],
        )
    assert runtime._lora_input is None


def test_comfyui_skips_synthetic_client_and_server_warmup():
    from sglang.multimodal_gen.runtime.server_warmup import (
        should_run_explicit_client_warmup,
        should_run_synthetic_server_warmup,
        SchedulerWarmupMixin,
    )

    args = SimpleNamespace(
        comfyui_mode=True, warmup_mode="server", warmup_resolutions=["448x256"]
    )
    assert not should_run_explicit_client_warmup(args)
    assert not should_run_synthetic_server_warmup(args)
    args.warmup_mode = "request"
    owner = SimpleNamespace(server_args=args, req_based_warmup_scheduled=False)
    requests = [(b"id", object())]
    assert (
        SchedulerWarmupMixin.process_received_reqs_with_req_based_warmup(
            owner, requests
        )
        is requests
    )


def test_spawned_workers_do_not_reexecute_launcher_main() -> None:
    """Workers must not re-run ComfyUI's main.py; its imports break under spawn."""
    import multiprocessing.spawn
    import sys

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
        _spawn_without_launcher_main,
    )

    main_dict = vars(sys.modules["__main__"])
    before = {k: main_dict.get(k, "<absent>") for k in ("__file__", "__spec__")}
    main_dict["__file__"] = "/comfy/main.py"
    try:
        with _spawn_without_launcher_main():
            data = multiprocessing.spawn.get_preparation_data("worker")
            assert "init_main_from_path" not in data
            assert "init_main_from_name" not in data
        assert main_dict["__file__"] == "/comfy/main.py"
    finally:
        for key, value in before.items():
            if value == "<absent>":
                main_dict.pop(key, None)
            else:
                main_dict[key] = value


def test_h3_vsa_backend_is_rejected_before_worker_load() -> None:
    """The integrated H3 step builds no VSA-H3 metadata; selecting it used to
    load the full model and only fail at the first sampling step."""
    import pytest

    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = lambda *a: (SimpleNamespace(), None, "minimax_h3")
    runtime.init_generator = lambda *a: (_ for _ in ()).throw(
        AssertionError("worker must not start")
    )
    for options in (
        {"attention_backend": "video_sparse_attn_h3"},
        {"component_attention_backends": "transformer=video_sparse_attn_h3"},
    ):
        with pytest.raises(ValueError, match="video_sparse_attn_h3 is not supported"):
            runtime.load_model(model_path="h3.gguf.missing", sgld_options=options)


def test_fasth3_single_file_as_base_h3_is_rejected_before_worker_load(
    tmp_path,
) -> None:
    """A FastH3 single file loaded as minimax_h3 used to start the worker and
    die on a raw state-dict mapping error for to_gate_compress."""
    import pytest
    import torch
    from safetensors.torch import save_file

    path = tmp_path / "fasth3.safetensors"
    save_file({"blocks.0.attn.to_gate_compress.weight": torch.zeros(1)}, path)
    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = lambda *a: (SimpleNamespace(), None, "minimax_h3")
    runtime.init_generator = lambda *a: (_ for _ in ()).throw(
        AssertionError("worker must not start")
    )
    with pytest.raises(ValueError, match="model_type fast_h3"):
        runtime.load_model(model_path=str(path), sgld_options={})


def test_fasth3_runtime_dir_is_checked_before_worker_load(tmp_path) -> None:
    """The 4-step preview and a raw (unmaterialized) FastH3 V2 download were
    only rejected by the worker after a full model load."""
    import json

    import pytest

    release = {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va"],
        "task_aliases": {},
        "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
    }
    (tmp_path / "transformer").mkdir()
    index = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    runtime = SGLDiffusionGenerator()
    runtime.get_comfyui_model = lambda *a: (SimpleNamespace(), None, "minimax_h3")
    runtime.init_generator = lambda *a: (_ for _ in ()).throw(
        AssertionError("worker must not start")
    )
    options = {"model_type": "fast_h3", "runtime_model_path": str(tmp_path)}
    for dmd, weight_map, message in (
        (None, {}, "no trained DMD rungs"),
        ([999, 500], {"blocks.0.x": "a.safetensors"}, "raw FastH3 download"),
    ):
        meta = dict(release, **({"dmd_denoising_steps": dmd} if dmd else {}))
        (tmp_path / "model_index.json").write_text(json.dumps({"_minimax_h3": meta}))
        index.write_text(json.dumps({"weight_map": weight_map}))
        with pytest.raises(ValueError, match=message):
            runtime.load_model(model_path="h3.safetensors", sgld_options=options)
