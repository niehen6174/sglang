# SPDX-License-Identifier: Apache-2.0

import pickle
from dataclasses import dataclass

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed import ipc_cuda
from sglang.multimodal_gen.runtime.distributed.ipc_cuda import (
    CudaIpcRef,
    _map_tree,
    attach_cuda_tensors,
    cuda_ipc_enabled,
    detach_cuda_tensors,
    materialize_cuda_refs,
    spill_cuda_tensors,
)


@pytest.fixture(autouse=True)
def _cuda_ipc_on(monkeypatch):
    # These tests exercise the IPC path; the platform default is covered below.
    monkeypatch.setenv("SGLANG_DIFFUSION_CUDA_IPC", "1")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA IPC tests")


def test_ipc_handle_pickle_is_much_smaller_than_tensor() -> None:
    _require_cuda()
    tensor = torch.ones(16, 90, 160, device="cuda", dtype=torch.bfloat16)
    ref = CudaIpcRef.from_tensor(tensor)
    assert len(pickle.dumps(ref)) < 2048
    rebuilt = ref.materialize()
    assert rebuilt.device.type == "cuda"
    assert rebuilt.shape == tensor.shape
    assert rebuilt.dtype == tensor.dtype
    assert torch.equal(rebuilt, tensor)


def test_spill_does_not_mutate_caller_req_tensors() -> None:
    _require_cuda()

    @dataclass
    class _Holder:
        latents: torch.Tensor
        prompt_embeds: list

    latents = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    embeds = [torch.ones(3, 5, device="cuda", dtype=torch.float32)]
    holder = _Holder(latents=latents, prompt_embeds=embeds)

    spilled = spill_cuda_tensors(holder)
    assert isinstance(spilled.latents, CudaIpcRef)
    assert isinstance(spilled.prompt_embeds[0], CudaIpcRef)
    assert holder.latents is latents
    assert holder.prompt_embeds[0] is embeds[0]

    restored = materialize_cuda_refs(spilled)
    assert torch.equal(restored.latents, latents)
    assert torch.equal(restored.prompt_embeds[0], embeds[0])


def test_detach_attach_keeps_non_cuda_fields() -> None:
    _require_cuda()

    @dataclass
    class _Holder:
        extra: dict
        vae_image_sizes: list
        image_embeds: list
        latents: torch.Tensor
        prompt_embeds: list

    latents = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    embeds = [torch.ones(2, 8, device="cuda", dtype=torch.float32)]
    holder = _Holder(
        extra={"comfyui_session_id": "run-1"},
        vae_image_sizes=[(64, 48)],
        image_embeds=[],
        latents=latents,
        prompt_embeds=embeds,
    )

    skeleton, tensors = detach_cuda_tensors(holder)
    assert holder.latents is latents
    assert skeleton.extra["comfyui_session_id"] == "run-1"
    assert skeleton.vae_image_sizes == [(64, 48)]
    assert skeleton.image_embeds == []
    assert skeleton.latents is None

    restored = attach_cuda_tensors(skeleton, tensors)
    assert restored.extra["comfyui_session_id"] == "run-1"
    assert restored.vae_image_sizes == [(64, 48)]
    assert restored.image_embeds == []
    assert restored.latents.device.type == "cuda"
    assert torch.equal(restored.latents, latents)
    assert torch.equal(restored.prompt_embeds[0], embeds[0])


def test_cross_process_ipc_roundtrip() -> None:
    _require_cuda()
    ctx = torch.multiprocessing.get_context("spawn")
    tensor = torch.arange(64, device="cuda", dtype=torch.float32).reshape(4, 16)
    ref = CudaIpcRef.from_tensor(tensor)
    queue = ctx.Queue()
    proc = ctx.Process(target=_cross_process_child, args=(queue, pickle.dumps(ref)))
    proc.start()
    try:
        shape, dtype, first, last = queue.get(timeout=30)
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join()
    assert proc.exitcode == 0
    assert shape == (4, 16)
    assert dtype == "torch.float32"
    assert first == 0.0
    assert last == 63.0


def _cross_process_child(queue, payload: bytes) -> None:
    ref = pickle.loads(payload)
    rebuilt = ref.materialize()
    queue.put(
        (
            tuple(rebuilt.shape),
            str(rebuilt.dtype),
            float(rebuilt.reshape(-1)[0]),
            float(rebuilt.reshape(-1)[-1]),
        )
    )


def test_tree_walk_reaches_tensors_nested_in_req_extra() -> None:
    """ControlNet residuals ride in req.extra as dict -> list -> tensor (None gaps)."""

    @dataclass
    class _Holder:
        extra: dict

    residual = torch.ones(1, 4, 8)
    holder = _Holder(
        extra={"comfyui_control": {"input": [residual, None], "output": [residual]}}
    )
    seen = []

    def mark(value):
        if isinstance(value, torch.Tensor):
            seen.append(value)
            return "ref"
        return value

    mapped = _map_tree(holder, mark, copy_dataclasses=True)
    assert len(seen) == 2
    assert mapped.extra["comfyui_control"] == {
        "input": ["ref", None],
        "output": ["ref"],
    }
    assert holder.extra["comfyui_control"]["input"][0] is residual


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("linux", None, True),
        ("win32", None, False),
        ("win32", "1", True),
        ("linux", "0", False),
        ("linux", "", True),
    ],
)
def test_cuda_ipc_platform_default(monkeypatch, platform, env, expected) -> None:
    monkeypatch.setattr(ipc_cuda.sys, "platform", platform)
    if env is None:
        monkeypatch.delenv("SGLANG_DIFFUSION_CUDA_IPC", raising=False)
    else:
        monkeypatch.setenv("SGLANG_DIFFUSION_CUDA_IPC", env)
    assert cuda_ipc_enabled() is expected


def test_without_ipc_tensors_pickle_through_host(monkeypatch) -> None:
    _require_cuda()
    monkeypatch.setenv("SGLANG_DIFFUSION_CUDA_IPC", "0")
    latents = torch.randn(1, 16, 8, 8, device="cuda", dtype=torch.bfloat16)
    batch = {"latents": latents, "timesteps": [torch.ones(1, device="cuda")]}
    spilled = spill_cuda_tensors(batch)
    assert spilled is batch and spilled["latents"] is latents
    restored = materialize_cuda_refs(pickle.loads(pickle.dumps(spilled)))
    assert restored["latents"].is_cuda
    assert torch.equal(restored["latents"], latents)
    assert torch.equal(restored["timesteps"][0], batch["timesteps"][0])
