# SPDX-License-Identifier: Apache-2.0
"""The diffusion runtime without the sgl_kernel extension or fcntl.

sgl_kernel publishes no Windows build and Windows has no fcntl; these paths are
what a Windows install runs.
"""

import sys

import pytest
import torch

from sglang import _file_lock
from sglang.kernels import fused_op
from sglang.srt.utils import device


class _Op(fused_op.BaseFusedOp):
    def forward_native(self, x):
        return "native"

    def forward_cuda(self, x):
        return "cuda"


@pytest.mark.parametrize(
    "has_sgl_kernel, expected", [(True, "cuda"), (False, "native")]
)
def test_cuda_forward_needs_sgl_kernel(monkeypatch, has_sgl_kernel, expected) -> None:
    monkeypatch.setattr(fused_op, "_oot_dispatch_key", lambda: None)
    monkeypatch.setattr(fused_op, "_platform_key", lambda: "cuda")
    monkeypatch.setattr(fused_op, "_has_sgl_kernel", lambda: has_sgl_kernel)
    assert _Op()(None) == expected


def test_sgl_kernel_availability_follows_the_install(monkeypatch) -> None:
    device.is_sgl_kernel_available.cache_clear()
    monkeypatch.setitem(sys.modules, "sgl_kernel", None)
    try:
        assert device.is_sgl_kernel_available() is False
    finally:
        device.is_sgl_kernel_available.cache_clear()


def test_file_lock_excludes_a_second_descriptor(tmp_path) -> None:
    path = tmp_path / "lock"
    with open(path, "w") as holder, open(path, "w") as other:
        _file_lock.flock(holder, _file_lock.LOCK_EX)
        with pytest.raises(BlockingIOError):
            _file_lock.flock(other, _file_lock.LOCK_EX | _file_lock.LOCK_NB)
        _file_lock.flock(holder, _file_lock.LOCK_UN)
        _file_lock.flock(other, _file_lock.LOCK_EX | _file_lock.LOCK_NB)
        _file_lock.flock(other, _file_lock.LOCK_UN)


def test_file_lock_shares_readers(tmp_path) -> None:
    path = tmp_path / "lock"
    with open(path, "w") as a, open(path, "w") as b:
        _file_lock.flock(a, _file_lock.LOCK_SH)
        _file_lock.flock(b, _file_lock.LOCK_SH | _file_lock.LOCK_NB)
        _file_lock.flock(a, _file_lock.LOCK_UN)
        _file_lock.flock(b, _file_lock.LOCK_UN)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm_without_sgl_kernel_matches_native(monkeypatch, with_residual) -> None:
    from sglang.multimodal_gen.runtime.layers import layernorm

    monkeypatch.setattr(layernorm, "_has_sgl_kernel", False)
    norm = layernorm.RMSNorm(3072, eps=1e-6, dtype=torch.bfloat16).cuda()
    torch.nn.init.normal_(norm.weight)
    x = torch.randn(2, 77, 3072, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None

    got = norm.forward_cuda(x.clone(), None if residual is None else residual.clone())
    want = norm.forward_native(
        x.clone(), None if residual is None else residual.clone()
    )
    if with_residual:
        for g, w in zip(got, want):
            assert g.shape == x.shape
            torch.testing.assert_close(g, w, rtol=0, atol=0)
    else:
        assert got.shape == x.shape
        torch.testing.assert_close(got, want, rtol=0, atol=0)
