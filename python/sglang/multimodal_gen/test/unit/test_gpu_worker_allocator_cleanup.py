"""Keep the CUDA allocator warm across ComfyUI's per-step model requests."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.managers import gpu_worker as module
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


@pytest.mark.parametrize(
    "kind,expected_clears",
    [
        ("noise", 0),
        ("decoded", 0),
        ("files", 1),
        ("empty", 1),
        ("raw", 0),
        ("error", 1),
    ],
)
def test_worker_reuses_allocator_for_noise_outputs(monkeypatch, kind, expected_clears):
    _run_request(monkeypatch, kind, expected_clears)


def _run_request(monkeypatch, kind, expected_clears, defer_release=False):
    monkeypatch.setattr(module, "trace_slice", lambda *args: nullcontext())
    device = Mock()
    monkeypatch.setattr(module.torch, "get_device_module", lambda: device)
    monkeypatch.setattr(module.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(module.current_platform, "is_mps", lambda: False)
    monkeypatch.setattr(module.current_platform, "is_cuda", lambda: True)
    worker = module.GPUWorker.__new__(module.GPUWorker)
    worker.is_output_rank = False
    worker.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(supports_auto_residency=False)
    )
    worker._release_warmup_pool = Mock()
    worker._realtime_sessions = SimpleNamespace(attach=Mock())
    worker._materialize_output_transport = Mock()
    worker._record_output_peak_memory = Mock()
    worker._deferred_finalize = None
    worker.defer_cache_release = defer_release
    worker._cache_release_due = None
    req = SimpleNamespace(
        is_warmup=False,
        extra={},
        metrics=None,
        trace_ctx=None,
        request_id="per-step",
        suppress_logs=True,
        return_raw_frames=kind == "raw",
        perf_dump_path=None,
        log=Mock(),
    )
    outputs = OutputBatch(
        noise_pred=[torch.zeros(1)] if kind == "noise" else None,
        output=[torch.zeros(1)] if kind == "decoded" else None,
        output_file_paths=["saved.mp4"] if kind == "files" else None,
    )

    def forward():
        if kind == "error":
            raise RuntimeError("failed forward")
        return outputs

    result = worker._execute_forward_common(
        req,
        forward_fn=forward,
        log_reqs=[req],
        return_req=False,
        save_output_paths=Mock(),
        error_context="test request",
    )

    assert device.empty_cache.call_count == expected_clears
    if kind == "error":
        assert "failed forward" in result.error
    else:
        assert result is outputs
        assert result.error is None
    return worker


def test_noise_outputs_do_not_schedule_idle_cache_release(monkeypatch):
    """Integrated mode sends one noise_pred reply per sampler step; the idle
    release must not fire between steps either."""
    worker = _run_request(monkeypatch, "noise", expected_clears=0, defer_release=True)
    assert worker._cache_release_due is None
