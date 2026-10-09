# SPDX-License-Identifier: Apache-2.0
"""One Qwen-Image 2.1 DiT forward per ComfyUI ``apply_model`` call.

ComfyUI's text encoder drops the vision tokens and reports ``image_slots``:
the text positions where each reference latent is spliced in. A placeholder
token at each slot turns that into the native ``build_layout`` input; the DiT
overwrites the placeholder rows with ``img_in(condition_latents)``. The
text / reference prefix K/V is cached per cond for the sampler run.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
    get_or_create_run_state,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)

# Must match COND_EXTRA_KEY in the plugin's executors/qwen_image21.py.
COND_EXTRA_KEY = "qwen21_cond"


def insert_image_placeholders(context, image_slots, num_refs):
    """Context with a zero token at each ComfyUI image slot, and the native flags."""
    text_len = context.shape[1]
    slots = [min(int(s), text_len) for s in image_slots][:num_refs]
    slots += [text_len] * (num_refs - len(slots))
    if slots != sorted(slots):
        raise ValueError(f"Qwen-Image 2.1 image_slots must be ascending, got {slots}")
    placeholder = context.new_zeros(context.shape[0], 1, context.shape[2])
    pieces, flags, start = [], [], 0
    for slot in slots:
        pieces += [context[:, start:slot], placeholder]
        flags += [False] * (slot - start) + [True]
        start = slot
    pieces.append(context[:, start:])
    flags += [False] * (text_len - start)
    return torch.cat(pieces, dim=1), flags


def _cache_fits(need_bytes: int, device) -> bool:
    free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else 0
    free = torch.tensor([free], dtype=torch.int64)
    # Ranks running one call must agree: under TP an uncached prefix runs
    # extra all-reduces, so a split decision would hang.
    if model_parallel_is_initialized():
        for group in (get_tp_group(), get_sp_group()):
            if group.world_size > 1:
                torch.distributed.all_reduce(
                    free, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
                )
    return int(free) > 2 * need_bytes


class QwenImage21ComfyUIStepStage(QwenImage21DenoisingStage):
    """ComfyUI owns the sampler loop; this runs the DiT once and returns the velocity."""

    def verify_input(self, batch, server_args) -> VerificationResult:
        return VerificationResult()

    def verify_output(self, batch, server_args) -> VerificationResult:
        return VerificationResult()

    def _new_cond(self, batch, latents) -> dict:
        embeds = batch.prompt_embeds
        if not (isinstance(embeds, list) and embeds):
            raise RuntimeError(
                "Qwen-Image 2.1 ComfyUI step has no conditioning for this cond; "
                "the worker lost its session (restarted mid-run?)"
            )
        arch = self.server_args.pipeline_config.dit_config.arch_config
        payload = (batch.extra or {}).get(COND_EXTRA_KEY) or {}
        bsz, _, height, width = latents.shape
        device, dtype = latents.device, torch.bfloat16
        context = embeds[0].to(device=device, dtype=dtype)
        context = context.expand(bsz, -1, -1)
        refs = [r for r in payload.get("ref_latents") or [] if r is not None]
        context, flags = insert_image_placeholders(
            context, payload.get("image_slots") or [], len(refs)
        )
        shapes = [(1, int(r.shape[-2]), int(r.shape[-1])) for r in refs]
        layout = build_layout(
            flags, shapes + [(1, height, width)], arch.axes_dims_rope, device
        )
        condition_latents = None
        if refs:
            condition_latents = torch.cat(
                [
                    r.to(device=device, dtype=dtype)
                    .expand(bsz, -1, -1, -1)
                    .flatten(2)
                    .transpose(1, 2)
                    for r in refs
                ],
                dim=1,
            )
        prefix_len = int(layout["prefix_rope"].shape[0])
        # K and V of every layer for every batch row, bf16.
        need = 2 * arch.num_layers * bsz * prefix_len * arch.hidden_size * 2
        caches = None
        if _cache_fits(need, device):
            caches = [[{} for _ in range(arch.num_layers)] for _ in range(bsz)]
        return dict(
            context=context,
            layout=layout,
            condition_latents=condition_latents,
            prefix_caches=caches,
        )

    def forward(self, batch, server_args):
        # Evicts older runs of this executor and restores dropped prompt_embeds.
        bind_comfyui_session(batch)
        latents = batch.latents
        bsz, _, height, width = latents.shape
        sp = get_sp_world_size()
        if (height * width) % sp:
            raise ValueError(
                f"Qwen-Image 2.1 sequence parallelism splits the {height}x{width} "
                f"latent ({height * width} tokens) across {sp} ranks, which needs a "
                f"token count divisible by {sp}; pick a width or height that is a "
                "multiple of 32 pixels, or run without sp_degree"
            )
        conds = get_or_create_run_state(batch, dict)
        key = ((batch.extra or {}).get("comfyui_cond_key"), bsz, height, width)
        if key not in conds:
            conds[key] = self._new_cond(batch, latents)
        cond = conds[key]
        timestep = batch.timesteps.to(device=latents.device, dtype=torch.float32)
        timestep = timestep.reshape(-1).expand(bsz).contiguous()
        hidden_states = latents.to(torch.bfloat16).flatten(2).transpose(1, 2)
        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ):
            output = self._predict_noise(
                current_model=self.transformer,
                latent_model_input=hidden_states.contiguous(),
                timestep=timestep,
                target_dtype=torch.bfloat16,
                guidance=None,
                encoder_hidden_states=cond["context"],
                layouts=[cond["layout"]] * bsz,
                condition_latents=cond["condition_latents"],
                prefix_caches=cond["prefix_caches"],
            )
        batch.noise_pred = output.transpose(1, 2).reshape(bsz, -1, height, width)
        return batch
