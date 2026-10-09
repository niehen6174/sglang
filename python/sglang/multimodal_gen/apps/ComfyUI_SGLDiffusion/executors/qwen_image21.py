# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 adapter for the ComfyUI DiT-forward contract.

ComfyUI calls ``diffusion_model(x, timestep, context, ref_latents=...,
image_slots=..., transformer_options=...)`` with ``x`` as ``[B, 64, H, W]``
latents (no patchify). The worker stage rebuilds ComfyUI's token order and
keeps the text / reference prefix K/V for the sampler run.
"""

from __future__ import annotations

import torch

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor

# Must match COND_EXTRA_KEY in the worker's qwen_image21_comfyui stage; not
# imported so the ComfyUI process does not load the denoising stack.
COND_EXTRA_KEY = "qwen21_cond"
CFG_PARALLEL_UNSUPPORTED = (
    "enable_cfg_parallel does not apply to Qwen-Image 2.1 in ComfyUI integrated "
    "mode: ComfyUI runs CFG itself and sends cond and uncond as separate (or "
    "batched) DiT calls, so every CFG rank would recompute the same call. Use "
    "sp_degree=2 (Ulysses) or tp_size=2 to spread one call across GPUs."
)

# Spatial downscale of the Qwen-Image 2.1 VAE; latents are not patchified.
_LATENT_SCALE = 16


def _last_model_sigma(sample_sigmas) -> float | None:
    if sample_sigmas is None:
        return None
    values = sample_sigmas.reshape(-1) if torch.is_tensor(sample_sigmas) else None
    if values is None or values.numel() < 2:
        return None
    nonzero = values[values > 0]
    return float(nonzero[-1]) if nonzero.numel() else None


def _prefix_cache_device(transformer_options) -> str:
    # QwenImage21Cache node: auto / gpu / cpu / off. Its int8 / int4 storage
    # dtype is not mirrored; the worker always keeps the lossless bf16 cache.
    options = (transformer_options or {}).get("qwen_image21_cache") or {}
    return str(options.get("device") or "auto")


def _hooks_present(transformer_options) -> bool:
    opts = transformer_options or {}
    patches = opts.get("patches") or {}
    replace = (opts.get("patches_replace") or {}).get("dit")
    return bool(
        replace
        or patches.get("post_input")
        or patches.get("single_block")
        or patches.get("attn1_patch")
    )


class QwenImage21Adapter(ComfyUIModelAdapter):
    model_types = ("qwen_image21",)
    pipeline_class_name = "QwenImage21Pipeline"

    def pack(
        self,
        x,
        timestep,
        context,
        ref_latents=None,
        image_slots=None,
        transformer_options=None,
        **kwargs,
    ) -> PackedForward:
        if _hooks_present(transformer_options):
            raise NotImplementedError(
                "Qwen-Image 2.1 SGLD integrated mode does not run ComfyUI DiT "
                "patches (model patches, Fun-Control, attention hooks); use UNETLoader"
            )
        bsz = int(x.shape[0])
        if context.shape[0] != bsz:
            context = context.expand(bsz, -1, -1)
        refs = [r for r in (ref_latents or []) if r is not None]
        payload = {
            "image_slots": [int(s) for s in (image_slots or [])],
            "ref_latents": refs,
            "prefix_cache": _prefix_cache_device(transformer_options),
            "last_sigma": _last_model_sigma(
                (transformer_options or {}).get("sample_sigmas")
            ),
        }
        return PackedForward(
            latents=x,
            timesteps=timestep.reshape(-1).float() * 1000.0,
            prompt_embeds=[context],
            prompt_seq_lens=[[int(context.shape[1])] * bsz],
            height=int(x.shape[-2]) * _LATENT_SCALE,
            width=int(x.shape[-1]) * _LATENT_SCALE,
            extra_req={COND_EXTRA_KEY: payload},
        )

    def unpack(self, noise_pred, packed, x):
        return noise_pred.to(device=x.device)

    def fill_req(self, req, packed: PackedForward) -> None:
        super().fill_req(req, packed)
        payload = packed.extra_req.get(COND_EXTRA_KEY)
        if payload is not None:
            extra = dict(req.extra or {})
            extra[COND_EXTRA_KEY] = payload
            req.extra = extra

    def drop_cached_fields(self, packed: PackedForward) -> None:
        super().drop_cached_fields(packed)
        packed.extra_req.pop(COND_EXTRA_KEY, None)


def qwen_image21_cond_key(packed: PackedForward) -> tuple | None:
    """Identify a cond batch by content, not just shape and two scalars.

    Positive and negative prompts share their chat-template head, so the
    first hidden state (and often the shape) match; the key also covers
    reference latents and slots, which ComfyUI's own prefix-cache key does.
    """
    embeds = packed.prompt_embeds
    if not embeds or not torch.is_tensor(embeds[0]) or embeds[0].numel() == 0:
        return None
    context = embeds[0]
    payload = packed.extra_req.get(COND_EXTRA_KEY) or {}
    parts = [context.float().reshape(context.shape[0], -1)]
    parts.extend(
        r.float().reshape(r.shape[0], -1).expand(context.shape[0], -1)
        for r in payload.get("ref_latents") or []
    )
    rows = torch.cat(parts, dim=1)
    weights = torch.linspace(1.0, 2.0, rows.shape[1], device=rows.device)
    stats = torch.stack(
        [rows.sum(1), rows.square().sum(1), (rows * weights).sum(1)], dim=1
    )
    return (
        tuple(int(d) for d in context.shape),
        tuple(payload.get("image_slots") or ()),
        tuple(tuple(int(d) for d in r.shape) for r in payload.get("ref_latents") or ()),
        tuple(round(float(v), 4) for v in stats.reshape(-1).tolist()),
    )


class QwenImage21Executor(SGLDiffusionExecutor):
    adapter_cls = QwenImage21Adapter

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)
        self.current_patcher = None

    @classmethod
    def validate_sgld_options(cls, sgld_options: dict | None) -> None:
        options = sgld_options or {}
        if (
            options.get("enable_cfg_parallel")
            or (options.get("cfg_parallel_degree") or 1) > 1
        ):
            raise ValueError(CFG_PARALLEL_UNSUPPORTED)

    def reset_prefix_cache(self, enabled):
        """ComfyUI's QwenImage21 calls this on pre_run / cleanup.

        The prefix K/V lives on the worker and is scoped to the sampler run
        (``comfyui_session_id``), so there is nothing to free here.
        """

    def _cond_key(self, packed) -> tuple | None:
        return qwen_image21_cond_key(packed)
