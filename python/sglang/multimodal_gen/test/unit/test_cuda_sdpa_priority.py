# SPDX-License-Identifier: Apache-2.0
"""SDPA kernel preference on CUDA builds with and without Flash SDPA.

The Windows CUDA wheels of PyTorch ship no Flash SDPA; there the default order
would pick the memory-efficient kernel instead of cuDNN.
"""

import pytest
import torch
from torch.nn.attention import SDPBackend

from sglang.multimodal_gen.runtime.platforms import cuda as cuda_platform


@pytest.fixture
def restore_sdpa_state():
    order = torch._C._get_sdp_priority_order()
    cudnn = torch.backends.cuda.cudnn_sdp_enabled()
    flash = torch.backends.cuda.flash_sdp_enabled()
    yield
    torch._C._set_sdp_priority_order(order)
    torch.backends.cuda.enable_cudnn_sdp(cudnn)
    torch.backends.cuda.enable_flash_sdp(flash)


def test_flash_build_keeps_cudnn_off(monkeypatch, restore_sdpa_state) -> None:
    monkeypatch.setattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: True
    )
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)
    order = torch._C._get_sdp_priority_order()
    cuda_platform._configure_cuda_sdpa()
    assert not torch.backends.cuda.cudnn_sdp_enabled()
    assert torch._C._get_sdp_priority_order() == order


def test_build_without_flash_prefers_cudnn(monkeypatch, restore_sdpa_state) -> None:
    monkeypatch.setattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: False
    )
    torch.backends.cuda.enable_cudnn_sdp(False)
    cuda_platform._configure_cuda_sdpa()
    assert torch.backends.cuda.cudnn_sdp_enabled()
    order = torch._C._get_sdp_priority_order()
    assert order[0] == int(SDPBackend.CUDNN_ATTENTION)
    assert order.index(int(SDPBackend.CUDNN_ATTENTION)) < order.index(
        int(SDPBackend.EFFICIENT_ATTENTION)
    )
