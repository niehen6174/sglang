# SPDX-License-Identifier: Apache-2.0
"""Wan 2.1 T2V and Wan 2.2 5B ComfyUI DiT adapter."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


class WanAdapter(ComfyUIModelAdapter):
    model_types = ("wan2.1",)
    pipeline_class_name = "WanPipeline"

    def pack(self, x, timestep, context, **kwargs):
        for name in (
            "clip_fea",
            "time_dim_concat",
            "reference_latent",
            "context_latents",
        ):
            if kwargs.get(name) is not None:
                raise ValueError(f"ComfyUI Wan does not yet support {name}")
        if x.ndim != 5 or x.shape[1] not in (16, 48):
            raise ValueError("ComfyUI Wan expects [B, 16 or 48, T, H, W] latents")
        h, w = x.shape[-2:]
        latents = F.pad(x, (0, w % 2, 0, h % 2, 0, 0), mode="circular")
        extra = {}
        if timestep.ndim == 2:
            if timestep.shape != (x.shape[0], x.shape[2]):
                raise ValueError(
                    "Wan frame timesteps must match the latent batch and frames"
                )
            # Wan's time embedding is per patch token, not per video frame.
            tokens_per_frame = (latents.shape[-2] // 2) * (latents.shape[-1] // 2)
            extra["extra"] = {
                "comfyui_model_timestep": timestep.repeat_interleave(
                    tokens_per_frame, dim=1
                )
            }
        spatial_scale = 16 if x.shape[1] == 48 else 8
        return PackedForward(
            latents=latents,
            timesteps=timestep.reshape(-1).max().reshape(1),
            prompt_embeds=[context],
            height=latents.shape[-2] * spatial_scale,
            width=latents.shape[-1] * spatial_scale,
            num_frames=(x.shape[2] - 1) * 4 + 1,
            extra_req=extra,
        )

    def unpack(self, noise_pred, packed, x):
        return noise_pred[:, :, : x.shape[2], : x.shape[3], : x.shape[4]].to(x.device)


class WanExecutor(SGLDiffusionExecutor):
    adapter_cls = WanAdapter

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)
        # WAN21.concat_cond inspects the patch convolution's channel count.
        self.patch_embedding = SimpleNamespace(
            weight=torch.empty((0, config.unet_config["in_dim"]), device="meta")
        )
