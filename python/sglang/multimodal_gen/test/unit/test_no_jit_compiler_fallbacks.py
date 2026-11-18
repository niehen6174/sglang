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


def test_jit_requires_the_device_compiler(tmp_path, monkeypatch) -> None:
    probe = toolchain.jit_unavailable_reason.__wrapped__
    nvcc = tmp_path / "nvcc"
    monkeypatch.setattr(toolchain.sys, "platform", "linux")
    monkeypatch.setattr(toolchain, "device_compiler_path", lambda: str(nvcc))
    assert "not found" in probe()
    nvcc.write_text("")
    assert probe() is None


def test_jit_is_refused_on_windows_even_with_nvcc(tmp_path, monkeypatch) -> None:
    # A CUDA toolkit on PATH used to enable JIT on Windows, where the POSIX
    # build rules then failed the first sampling step.
    probe = toolchain.jit_unavailable_reason.__wrapped__
    nvcc = tmp_path / "nvcc.exe"
    nvcc.write_text("")
    monkeypatch.setattr(toolchain.sys, "platform", "win32")
    monkeypatch.setattr(toolchain, "device_compiler_path", lambda: str(nvcc))
    assert "native Windows" in probe()


def test_can_build_jit_follows_the_reason(monkeypatch) -> None:
    probe = toolchain.can_build_jit.__wrapped__
    monkeypatch.setattr(toolchain, "jit_unavailable_reason", lambda: "no nvcc")
    assert probe() is False
    monkeypatch.setattr(toolchain, "jit_unavailable_reason", lambda: None)
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

    rotary_emb = minimax_h3.RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rope_dim,
        is_neox_style=True,
        use_precomputed_cache=False,
    )

    def apply(q, k):
        return minimax_h3._apply_rope_qk(q, k, cache, positions, rotary_emb=rotary_emb)

    want_q, want_k = apply(q.clone(), k.clone())
    monkeypatch.setattr(minimax_h3, "_has_sgl_kernel", lambda: False)
    got_q, got_k = apply(q.clone(), k.clone())

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


@cuda
def test_convrot_int8_jit_reports_the_toolchain_reason(monkeypatch) -> None:
    from sglang.multimodal_gen.runtime.layers.quantization import convrot_int8_jit

    capability = convrot_int8_jit.convrot_int8_local_capability()
    if capability not in convrot_int8_jit.convrot_int8_supported_capabilities():
        pytest.skip("convrot_int8 JIT does not target this GPU")
    probe = convrot_int8_jit.jit_convrot_unavailable_reason.__wrapped__
    monkeypatch.setattr(convrot_int8_jit, "jit_unavailable_reason", lambda: "no jit")

    def build():
        raise AssertionError("must not attempt a build")

    monkeypatch.setattr(convrot_int8_jit, "load_convrot_int8_module", build)
    assert probe() == "no jit"


def test_load_jit_refuses_without_running_a_compiler(monkeypatch) -> None:
    from sglang.kernels.jit.utils.compile import cache, loader

    monkeypatch.setattr(toolchain, "jit_unavailable_reason", lambda: "no compiler")

    def fingerprint(*args, **kwargs):
        raise AssertionError("must not fingerprint or build")

    monkeypatch.setattr(cache, "compute_build_key", fingerprint)
    with pytest.raises(RuntimeError, match="no compiler"):
        loader.load_jit("probe_kernel", cuda_files=["probe.cuh"])


@cuda
def test_h3_comfy_kitchen_qk_norm_rope_matches_eager(monkeypatch) -> None:
    pytest.importorskip("comfy_kitchen")
    if not minimax_h3._comfy_kitchen_qknorm_rope():
        pytest.skip("comfy_kitchen has no CUDA backend here")
    tokens, heads, head_dim, rope_dim = 300, 4, 128, 96
    torch.manual_seed(0)
    freqs = torch.randn(tokens, rope_dim, device="cuda")
    cache = minimax_h3._rope_cos_sin_cache(freqs, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda")
    # Q/K as strided views of a packed QKV buffer, as the attention layer has them.
    qkv = torch.randn(tokens, 3, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_norm = torch.nn.RMSNorm(head_dim, eps=1e-6, device="cuda", dtype=torch.bfloat16)
    k_norm = torch.nn.RMSNorm(head_dim, eps=1e-6, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        q_norm.weight.uniform_(0.5, 1.5)
        k_norm.weight.uniform_(0.5, 1.5)
        monkeypatch.setattr(minimax_h3, "_has_sgl_kernel", lambda: False)
        want_q, want_k = minimax_h3._apply_qk_norm_rope(
            qkv[:, 0].clone(),
            qkv[:, 1].clone(),
            q_norm,
            k_norm,
            head_dim,
            cache,
            positions,
            minimax_h3.RotaryEmbedding(
                head_size=head_dim,
                rotary_dim=rope_dim,
                is_neox_style=True,
                use_precomputed_cache=False,
            ),
        )
        q, k = qkv[:, 0], qkv[:, 1]
        assert minimax_h3._comfy_kitchen_qk_norm_rope_(
            q, k, q_norm, k_norm, cache, positions
        )
    torch.testing.assert_close(q, want_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k, want_k, rtol=2e-2, atol=2e-2)
    # V is untouched by the in-place kernel.
    assert minimax_h3._comfy_kitchen_rope_table(cache, positions) is (
        minimax_h3._comfy_kitchen_rope_table(cache, positions)
    )
