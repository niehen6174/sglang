# SPDX-License-Identifier: Apache-2.0
"""Diffusion kernels with an eager or Triton equivalent run without a JIT compiler.

A runtime without the CUDA toolkit (the Windows CUDA wheels of PyTorch) cannot
build JIT kernels and has no sgl_kernel extension.
"""

import pytest
import torch

from sglang.kernels.jit.utils.compile import toolchain
from sglang.kernels.kda_kernels import residual_gate_add_jit
from sglang.multimodal_gen.runtime.models.dits import minimax_h3


def test_can_build_jit_requires_the_device_compiler(tmp_path, monkeypatch) -> None:
    probe = toolchain.can_build_jit.__wrapped__
    nvcc = tmp_path / "nvcc"
    monkeypatch.setattr(toolchain, "device_compiler_path", lambda: str(nvcc))
    assert probe() is False
    nvcc.write_text("")
    assert probe() is True


def test_windows_compiler_name_is_accepted(tmp_path, monkeypatch) -> None:
    probe = toolchain.can_build_jit.__wrapped__
    monkeypatch.setattr(
        toolchain, "device_compiler_path", lambda: str(tmp_path / "nvcc")
    )
    (tmp_path / "nvcc.exe").write_text("")
    assert probe() is True


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_residual_gate_add_stays_eager_without_compiler(monkeypatch) -> None:
    residual = torch.randn(2, 64, 256, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(2, 1, 256, device="cuda", dtype=torch.bfloat16)
    assert residual_gate_add_jit._route(residual, update, gate) is not None
    monkeypatch.setattr(residual_gate_add_jit, "can_build_jit", lambda: False)
    assert residual_gate_add_jit._route(residual, update, gate) is None
    torch.testing.assert_close(
        residual_gate_add_jit.residual_gate_add(residual, update, gate),
        residual + update * gate,
        rtol=0,
        atol=0,
    )


@cuda
def test_h3_eager_rope_matches_sgl_kernel(monkeypatch) -> None:
    pytest.importorskip("sgl_kernel")
    tokens, heads, head_dim, rope_dim = 37, 4, 128, 96
    torch.manual_seed(0)
    freqs = torch.randn(64, rope_dim, device="cuda")
    cache = minimax_h3._rope_cos_sin_cache(freqs, dtype=torch.bfloat16)
    positions = torch.randint(0, 64, (tokens,), device="cuda")
    q = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)

    want_q, want_k = minimax_h3._apply_rope_qk(q.clone(), k.clone(), cache, positions)
    monkeypatch.setattr(minimax_h3, "_has_sgl_kernel", lambda: False)
    got_q, got_k = minimax_h3._apply_rope_qk(q.clone(), k.clone(), cache, positions)

    # sgl_kernel rotates in FP32 and rounds once; eager rounds per op in BF16.
    torch.testing.assert_close(got_q, want_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got_k, want_k, rtol=2e-2, atol=2e-2)
    # The head dims past the rotary width pass through untouched.
    torch.testing.assert_close(got_q[..., rope_dim:], q[..., rope_dim:])


@cuda
def test_h3_silu_mul_eager_without_compiler(monkeypatch) -> None:
    hidden = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(minimax_h3, "can_build_jit", lambda: False)
    gate, up = hidden.chunk(2, dim=-1)
    torch.testing.assert_close(
        minimax_h3._silu_mul(hidden, reuse_input=False),
        torch.nn.functional.silu(gate) * up,
        rtol=0,
        atol=0,
    )
