"""Comfy NVFP4 block scales are serialized swizzled in 128x4 tiles.

Tensor-parallel loading used to narrow that swizzled tensor directly, which is
only correct along output rows; an input-dim (row-parallel) shard mixed scales
across blocks and the folded H3 text encoder produced unrelated videos.
"""

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization.comfy_nvfp4 import (
    _linear_nvfp4_scales_to_swizzled,
    _shard_swizzled_scales,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    _swizzled_nvfp4_scales_to_linear,
)


def _linear_scales(rows=256, cols=16):
    return torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)


def test_swizzle_round_trip():
    linear = _linear_scales()
    swizzled = _linear_nvfp4_scales_to_swizzled(linear)
    assert not torch.equal(swizzled, linear)
    assert torch.equal(_swizzled_nvfp4_scales_to_linear(swizzled), linear)


def _narrowing_loader(dim, rank, world):
    def load(param, loaded):
        size = param.data.shape[dim]
        param.data.copy_(loaded.narrow(dim, rank * size, size))

    return load


def test_tp_shards_match_the_unsharded_scales():
    linear = _linear_scales()
    checkpoint = _linear_nvfp4_scales_to_swizzled(linear)
    for dim in (0, 1):
        for rank in range(2):
            shape = list(linear.shape)
            shape[dim] //= 2
            param = nn.Parameter(torch.empty(shape), requires_grad=False)
            _shard_swizzled_scales(_narrowing_loader(dim, rank, 2))(param, checkpoint)
            expected = linear.narrow(dim, rank * shape[dim], shape[dim])
            assert torch.equal(_swizzled_nvfp4_scales_to_linear(param.data), expected)
