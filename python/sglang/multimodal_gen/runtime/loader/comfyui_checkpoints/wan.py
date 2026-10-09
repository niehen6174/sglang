# SPDX-License-Identifier: Apache-2.0
"""Wan ComfyUI checkpoints retain the original Wan parameter names."""

import re

from safetensors import safe_open

from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)

from .spec import ComfyUICheckpointSpec, register_comfyui_checkpoint


def _build_dit_config(server_args):
    with safe_open(server_args.model_path, framework="pt", device="cpu") as checkpoint:
        prefix = (
            "model.diffusion_model."
            if "model.diffusion_model.patch_embedding.weight" in checkpoint.keys()
            else ""
        )
        keys = {key.removeprefix(prefix) for key in checkpoint.keys()}

        def shape(name):
            return checkpoint.get_slice(prefix + name).get_shape()

        dim, channels, *patch = shape("patch_embedding.weight")
        out_channels = shape("head.head.weight")[0] // (patch[0] * patch[1] * patch[2])
        layers = {
            int(m.group(1)) for key in keys if (m := re.match(r"blocks\.(\d+)\.", key))
        }
        if channels != out_channels or channels not in (16, 48):
            raise ValueError(
                "ComfyUI Wan currently supports Wan 2.1 T2V and Wan 2.2 5B only"
            )
        allowed = (
            "patch_embedding.",
            "text_embedding.",
            "time_embedding.",
            "time_projection.",
            "blocks.",
            "head.",
        )
        if any(not key.startswith(allowed) for key in keys):
            raise ValueError(
                "Unsupported Wan checkpoint variant or serialized quantization"
            )
        arch = WanVideoArchConfig(
            num_attention_heads=dim // 128,
            attention_head_dim=128,
            in_channels=channels,
            out_channels=out_channels,
            patch_size=tuple(patch),
            freq_dim=shape("time_embedding.0.weight")[1],
            text_dim=shape("text_embedding.0.weight")[1],
            ffn_dim=shape("blocks.0.ffn.0.weight")[0],
            num_layers=len(layers),
        )
    config = WanVideoConfig(arch_config=arch)
    server_args.pipeline_config.dit_config = config
    server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial = (
        16 if channels == 48 else 8
    )
    server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal = 4
    return config


_MAPPING = {
    r"^model\.diffusion_model\.(.*)$": (r"\1", None, None),
    r"^patch_embedding\.(.*)$": (r"patch_embedding.proj.\1", None, None),
    r"^text_embedding\.0\.(.*)$": (
        r"condition_embedder.text_embedder.fc_in.\1",
        None,
        None,
    ),
    r"^text_embedding\.2\.(.*)$": (
        r"condition_embedder.text_embedder.fc_out.\1",
        None,
        None,
    ),
    r"^time_embedding\.0\.(.*)$": (
        r"condition_embedder.time_embedder.mlp.fc_in.\1",
        None,
        None,
    ),
    r"^time_embedding\.2\.(.*)$": (
        r"condition_embedder.time_embedder.mlp.fc_out.\1",
        None,
        None,
    ),
    r"^time_projection\.1\.(.*)$": (
        r"condition_embedder.time_modulation.linear.\1",
        None,
        None,
    ),
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": (r"blocks.\1.to_out.\2", None, None),
    r"^blocks\.(\d+)\.self_attn\.(q|k|v)\.(.*)$": (r"blocks.\1.to_\2.\3", None, None),
    r"^blocks\.(\d+)\.self_attn\.norm_(q|k)\.(.*)$": (
        r"blocks.\1.norm_\2.\3",
        None,
        None,
    ),
    r"^blocks\.(\d+)\.cross_attn\.(q|k|v)\.(.*)$": (
        r"blocks.\1.attn2.to_\2.\3",
        None,
        None,
    ),
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": (
        r"blocks.\1.attn2.to_out.\2",
        None,
        None,
    ),
    r"^blocks\.(\d+)\.cross_attn\.norm_(q|k)\.(.*)$": (
        r"blocks.\1.attn2.norm_\2.\3",
        None,
        None,
    ),
    r"^blocks\.(\d+)\.norm3\.(.*)$": (
        r"blocks.\1.self_attn_residual_norm.norm.\2",
        None,
        None,
    ),
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": (r"blocks.\1.ffn.fc_in.\2", None, None),
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": (r"blocks.\1.ffn.fc_out.\2", None, None),
    r"^blocks\.(\d+)\.modulation$": (r"blocks.\1.scale_shift_table", None, None),
    r"^head\.head\.(.*)$": (r"proj_out.\1", None, None),
    r"^head\.modulation$": (r"scale_shift_table", None, None),
}
register_comfyui_checkpoint(
    "WanPipeline",
    ComfyUICheckpointSpec(
        dit_cls_name="WanTransformer3DModel",
        build_dit_config=_build_dit_config,
        param_names_mapping=_MAPPING,
        inherit_config_mapping=True,
    ),
)
