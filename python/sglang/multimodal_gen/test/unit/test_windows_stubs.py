# SPDX-License-Identifier: Apache-2.0
"""sglang/_windows_stubs.py: what a Windows interpreter gets in place of
fcntl, resource and sgl_kernel. The fcntl stand-in calls Win32 APIs and is
only exercised on Windows; the rest runs anywhere in a fresh interpreter."""

import subprocess
import sys
import textwrap

import pytest
import torch


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_sgl_kernel_stub_imports_and_refuses_calls() -> None:
    out = _run("""
        import os, sys
        sys.modules["sgl_kernel"] = None  # as on Windows: not installed
        os.environ.pop("SGLANG_FORCE_FUSED_OP_BACKEND", None)
        from sglang import _windows_stubs
        _windows_stubs.install_windows_stubs()

        import sgl_kernel
        from sgl_kernel import rmsnorm
        from sgl_kernel.quantization import ggml_dequantize
        print(sgl_kernel.is_stub, _windows_stubs.sgl_kernel_available())
        print(os.environ["SGLANG_FORCE_FUSED_OP_BACKEND"])
        try:
            rmsnorm(1)
        except RuntimeError as e:
            print("raises", "sgl_kernel.rmsnorm" in str(e))
        try:
            import sgl_kernel.scalar_type
        except ImportError:
            print("scalar_type missing")
        """)
    assert out.split("\n")[:4] == [
        "True False",
        "torch",
        "raises True",
        "scalar_type missing",
    ]


def test_resource_stub_makes_limit_raising_a_no_op() -> None:
    out = _run("""
        import sys
        sys.modules["resource"] = None  # as on Windows: no such module
        from sglang import _windows_stubs
        _windows_stubs.install_windows_stubs()

        import resource
        from sglang.srt.utils.process import set_ulimit
        set_ulimit()
        print(resource.getrlimit(resource.RLIMIT_NOFILE)[0] == resource.RLIM_INFINITY)
        """)
    assert out.strip() == "True"


def test_installed_modules_are_left_alone() -> None:
    out = _run("""
        import fcntl, resource
        from sglang import _windows_stubs
        _windows_stubs.install_windows_stubs()
        import sys
        print(sys.modules["fcntl"] is fcntl, sys.modules["resource"] is resource)
        """)
    assert out.strip() == "True True"


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
