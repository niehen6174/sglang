# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_block
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64
MINIMAX_H3_ADALN_MODALITY_NUM = 3


@dataclass
class MiniMaxH3DiTArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_block])

    lora_param_names_mapping: dict = field(default_factory=dict)

    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.AITER,
            AttentionBackendEnum.TORCH_SDPA,
        }
    )

    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # Comfy MiniMax H3 checkpoints store qkv rows as [Q_all, K_all, V_all].
    # Some other checkpoints use grouped [q,k,v] per query group instead.
    qkv_checkpoint_layout: str = "flat"
    # When set (typically 1025), the pruned Comfy checkpoint stores a shared
    # time-embedding curve table instead of ``time_embedder`` weights.
    adaln_curve_grid: int | None = None
    adaln_curve_dim: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values, got {self.patch_size}.")
        if self.adaln_curve_grid is not None and self.adaln_curve_grid < 2:
            raise ValueError(
                "adaln_curve_grid must be at least 2 when curve pruning is enabled."
            )
        if self.adaln_curve_dim <= 0:
            raise ValueError("adaln_curve_dim must be positive.")
        self.num_channels_latents = self.latents_dim

    @property
    def use_adaln_curve(self) -> bool:
        return self.adaln_curve_grid is not None

    @property
    def adaln_input_dim(self) -> int:
        return self.adaln_curve_dim if self.use_adaln_curve else self.time_embed_dim


@dataclass
class MiniMaxH3DiTConfig(DiTConfig):
    arch_config: MiniMaxH3DiTArchConfig = field(default_factory=MiniMaxH3DiTArchConfig)


__all__ = [
    "MINIMAX_H3_ADALN_MODALITY_NUM",
    "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT",
    "MiniMaxH3DiTArchConfig",
    "MiniMaxH3DiTConfig",
]
