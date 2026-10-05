"""Avoid copying fused QKV views while respecting packed attention boundaries."""

import importlib

import pytest
import torch

from sglang.multimodal_gen.runtime.managers import gpu_worker  # noqa: F401


def make_impl(backend, heads, dim, padding=True):
    if backend == "sage2":
        pytest.importorskip("sageattention")
        module = importlib.import_module(
            "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn"
        )
        return module.SageAttentionImpl(
            heads, dim, False, dim**-0.5, packed_trailing_padding=padding
        )
    pytest.importorskip("sageattn3")
    module = importlib.import_module(
        "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3"
    )
    return module.SageAttention3Impl(
        heads, dim, False, dim**-0.5, packed_trailing_padding=padding
    )


@pytest.mark.parametrize("backend", ["sage2", "sage3"])
@pytest.mark.parametrize("last_stride", [1, 2])
@pytest.mark.parametrize("padding", [True, False])
def test_packed_qkv_preserves_legal_views(monkeypatch, backend, last_stride, padding):
    fused = torch.arange(8 * 3 * 2 * 8 * last_stride, dtype=torch.float32).reshape(
        8, 3, 2, 8 * last_stride
    )
    parts = tuple(fused[:, i, :, ::last_stride] for i in range(3))
    impl = make_impl(backend, 2, 8, padding)
    bounds = (0, 5, 8)
    spans = [(0, 5)] if padding else [(0, 5), (5, 8)]
    calls = []

    def kernel(q, k, v, metadata):
        start, stop = spans[len(calls)]
        for got, original in zip([q, k, v], parts):
            expected = original[start:stop].unsqueeze(0)
            if last_stride == 1:
                assert got.data_ptr() == expected.data_ptr()
                assert got.stride() == expected.stride()
            else:
                assert got.stride(-1) == 1
                assert torch.equal(got, expected)
        calls.append((start, stop))
        return q + q.mean(dim=1, keepdim=True)

    monkeypatch.setattr(impl, "forward", kernel)
    out = impl.forward_varlen(
        *parts,
        cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
        max_seqlen=5,
        cu_seqlens_host=bounds,
    )
    assert calls == spans
    for start, stop in spans:
        assert torch.equal(
            out[start:stop],
            parts[0][start:stop] + parts[0][start:stop].mean(dim=0, keepdim=True),
        )
    if padding:
        assert torch.count_nonzero(out[5:]) == 0


@pytest.mark.parametrize("backend", ["sage2", "sage3"])
@pytest.mark.parametrize("used", [257, 1025])
@pytest.mark.parametrize("last_stride", [1, 2])
def test_gpu_packed_strided_matches_contiguous(backend, used, last_stride):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("This regression covers the installed Blackwell kernels")
    torch.manual_seed(341)
    total = ((used + 63) // 64) * 64
    fused = torch.randn(
        total, 3, 4, 128 * last_stride, device="cuda", dtype=torch.bfloat16
    )
    parts = tuple(fused[:, i, :, ::last_stride] for i in range(3))
    impl = make_impl(backend, 4, 128)
    reference = impl.forward(
        *[p[:used].contiguous().unsqueeze(0) for p in parts], None
    )[0]
    result = impl.forward_varlen(
        *parts,
        cu_seqlens=torch.tensor([0, used, total], device="cuda", dtype=torch.int32),
        max_seqlen=used,
        cu_seqlens_host=(0, used, total),
    )
    assert bool(torch.isfinite(result).all())
    assert torch.equal(result[:used], reference), (
        f"{backend}: strided input changed the output"
    )
    assert torch.count_nonzero(result[used:]) == 0
