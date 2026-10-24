# SPDX-License-Identifier: Apache-2.0
"""Snapshot allocator ownership across real GPUWorker forward/finalize methods."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.runtime.managers import gpu_worker as worker_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    component_manager as manager_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
    ComponentUse,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    SnapshotOffloadStrategy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


@pytest.fixture
def forward_worker(monkeypatch):
    device = SimpleNamespace(
        empty_cache=Mock(),
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: None,
        Event=lambda: SimpleNamespace(record=lambda: None),
    )
    monkeypatch.setattr(worker_module.torch, "get_device_module", lambda: device)
    platform = SimpleNamespace(
        is_cpu=lambda: False,
        is_mps=lambda: False,
        is_cuda=lambda: True,
        is_npu=lambda: False,
        device_shares_host_memory=lambda: False,
    )
    monkeypatch.setattr(worker_module, "current_platform", platform)
    monkeypatch.setattr(manager_module, "current_platform", platform)
    monkeypatch.setattr(worker_module, "trace_slice", lambda *args: nullcontext())
    args = SimpleNamespace(
        performance_mode="manual",
        enable_layerwise_nvtx_marker=False,
        pipeline_config=SimpleNamespace(supports_auto_residency=False),
    )
    manager = ComponentResidencyManager(
        SimpleNamespace(
            modules={}, _stage_name_mapping={}, component_residency_strategies={}
        ),
        args,
    )
    monkeypatch.setattr(manager, "_keep_ready_component_on_cuda", lambda: False)
    monkeypatch.setattr(manager, "_module_on_supported_device", lambda _: False)
    worker = GPUWorker.__new__(GPUWorker)
    worker.server_args = args
    worker.pipeline = SimpleNamespace(component_residency_manager=manager)
    worker.defer_cache_release = True
    worker._cache_release_due = None
    worker.is_output_rank = False
    worker._release_warmup_pool = lambda _: None
    worker._realtime_sessions = SimpleNamespace(attach=lambda _: None)
    worker._to_output_batch = lambda result: result
    worker._iter_output_metrics = lambda _: []
    worker._can_defer_finalize = lambda _: True
    worker._finalize_deferred = Mock()
    req = SimpleNamespace(
        is_warmup=False,
        return_raw_frames=False,
        extra={},
        metrics=None,
        trace_ctx=None,
        log=lambda **kwargs: None,
    )
    return worker, manager, req, device


def _forward(worker, req, forward, *, return_req=False, propagate=False):
    return worker._execute_forward_common(
        req,
        forward_fn=forward,
        log_reqs=[req],
        return_req=return_req,
        save_output_paths=lambda _: None,
        error_context="test",
        defer_finalize=True,
        propagate_forward_errors=propagate,
    )


def _released(manager):
    manager._empty_cache_after_large_release(
        ComponentUse("DiT", "transformer", memory_intensive=True),
        SnapshotOffloadStrategy(),
        object(),
        True,
    )


def test_deferred_finalize_captures_its_own_pending_value(forward_worker):
    worker, manager, req, device = forward_worker

    def forward():
        _released(manager)
        return OutputBatch(output=[object()])

    _forward(worker, req, forward)
    previous = worker._deferred_finalize
    assert previous.keywords["snapshot_cache_release_pending"] is True
    assert manager._snapshot_cache_release_request is None
    _forward(worker, req, lambda: OutputBatch(output=[object()]))
    assert worker._deferred_finalize.keywords["snapshot_cache_release_pending"] is False
    assert previous.keywords["snapshot_cache_release_pending"] is True
    previous()
    assert (
        worker._finalize_deferred.call_args.kwargs["snapshot_cache_release_pending"]
        is True
    )
    device.empty_cache.assert_not_called()


@pytest.mark.parametrize("mode", ["no-idle", "warmup", "raw", "disaggregated"])
def test_forward_without_idle_ownership_keeps_eager_release(forward_worker, mode):
    worker, manager, req, device = forward_worker
    worker.defer_cache_release = mode != "no-idle"
    req.is_warmup = mode == "warmup"
    req.return_raw_frames = mode == "raw"

    def forward():
        _released(manager)
        return OutputBatch(output=[object()])

    _forward(worker, req, forward, return_req=mode == "disaggregated")
    device.empty_cache.assert_called_once()
    assert worker._deferred_finalize.keywords["snapshot_cache_release_pending"] is False


def test_propagated_forward_error_still_schedules_idle_cleanup(forward_worker):
    worker, manager, req, device = forward_worker

    def forward():
        _released(manager)
        raise ValueError("forward failed")

    with pytest.raises(ValueError, match="forward failed"):
        _forward(worker, req, forward, propagate=True)
    assert worker._cache_release_due is not None
    assert manager._snapshot_cache_release_request is None
    device.empty_cache.assert_not_called()
