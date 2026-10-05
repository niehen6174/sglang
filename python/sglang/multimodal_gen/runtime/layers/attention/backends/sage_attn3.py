# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from sageattn3 import sageattn3_blackwell

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _kernel_input(value: torch.Tensor) -> torch.Tensor:
    """Preserve fused QKV views supported by Sage's stride-aware kernels."""
    return value if value.stride(-1) == 1 else value.contiguous()


class SageAttention3Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN_3

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError


class SageAttention3Impl(AttentionImpl):
    _warned_gqa_fallback_global: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        self.packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # SageAttention3's Blackwell kernel assumes MHA (Hq == Hkv). For GQA/MQA
        # (Hq != Hkv), fall back to torch SDPA which supports GQA.
        if key.shape[1] != query.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA/MQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}"
                )
            if not type(self)._warned_gqa_fallback_global:
                logger.warning(
                    "SageAttention3 does not support GQA/MQA (Hq != Hkv); falling back to torch SDPA."
                )
                type(self)._warned_gqa_fallback_global = True
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=self.causal,
                dropout_p=self.dropout,
                scale=self.softmax_scale,
                enable_gqa=True,
            )
        else:
            output = sageattn3_blackwell(query, key, value, is_causal=self.causal)
        output = output.transpose(1, 2)
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Run the real Blackwell kernel independently on each packed sequence."""
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(x) for x in cu_seqlens.tolist())
        )
        used = (
            trailing_padding_used_len(query.shape[0], max_seqlen, bounds)
            if self.packed_trailing_padding
            else None
        )
        output = torch.zeros_like(query)
        spans = ((0, used),) if used is not None else zip(bounds[:-1], bounds[1:])
        for start, stop in spans:
            if start == stop:
                continue
            output[start:stop] = self.forward(
                _kernel_input(query[start:stop]).unsqueeze(0),
                _kernel_input(key[start:stop]).unsqueeze(0),
                _kernel_input(value[start:stop]).unsqueeze(0),
                None,
            )[0]
        return output
