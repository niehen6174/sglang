# SPDX-License-Identifier: Apache-2.0
"""One Qwen-Image 2.1 DiT forward per ComfyUI ``apply_model`` call.

ComfyUI's ``QwenImage21Transformer2DModel`` drops the vision tokens from the
text sequence and records ``image_slots`` (insertion points); reference
latents are spliced in at those slots, the target image comes last, and the
text/reference prefix is block-causal with t = 0 modulation. This stage
rebuilds that sequence as the native DiT's ``layout`` and keeps the prefix
K/V per conditioning for one sampler run, like ComfyUI's prefix cache.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    begin_comfyui_run,
    get_or_create_run_state,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

COND_EXTRA_KEY = "qwen21_cond"
_ROPE_THETA = 10000


def _grid_offsets(size: int, target_size: int) -> torch.Tensor:
    # Matches ComfyUI build_sequence: a reference grid whose parity differs
    # from the target shifts by half a token so both centre on the same point.
    return (
        torch.arange(size, dtype=torch.float64)
        - (size - size // 2)
        + 0.5 * (size % 2 - target_size % 2)
    )


def _grid_positions(t_pos: int, h: int, w: int, target_hw) -> torch.Tensor:
    hh = _grid_offsets(h, target_hw[0])
    ww = _grid_offsets(w, target_hw[1])
    return torch.stack(
        [
            torch.full((h, w), float(t_pos), dtype=torch.float64),
            hh[:, None].expand(h, w),
            ww[None, :].expand(h, w),
        ],
        dim=-1,
    ).reshape(-1, 3)


def _rope_table(positions: torch.Tensor, axes_dims) -> torch.Tensor:
    # ComfyUI's flux rope: fp64 frequencies, cos/sin rounded to fp32.
    angles = []
    for axis, dim in enumerate(axes_dims):
        scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
        omega = 1.0 / (_ROPE_THETA**scale)
        angles.append(positions[:, axis : axis + 1] * omega[None, :])
    angles = torch.cat(angles, dim=-1)
    return torch.complex(angles.cos().float(), angles.sin().float())


def build_comfyui_layout(
    *,
    text_len: int,
    image_slots: list[int],
    ref_shapes: list[tuple[int, int]],
    target_hw: tuple[int, int],
    axes_dims,
    device,
) -> dict[str, Any]:
    """The native DiT layout for ComfyUI's ``build_sequence`` token order.

    Reference rows point at text index 0; the DiT overwrites them with
    ``img_in(condition_latents)`` through ``image_indices``.
    """
    slots = (list(image_slots) + [text_len] * len(ref_shapes))[: len(ref_shapes)]
    bounds = [0, *slots, text_len]
    text_indices: list[torch.Tensor] = []
    image_indices: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    segments: list[tuple[int, int, bool]] = []
    length = 0
    pos = 0
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        n = end - start
        if n > 0:
            text_indices.append(torch.arange(start, end))
            positions.append(
                torch.arange(pos, pos + n, dtype=torch.float64)[:, None].expand(n, 3)
            )
            segments.append((length, length + n, False))
            pos += n
            length += n
        if i == len(ref_shapes):
            break
        h, w = ref_shapes[i]
        text_indices.append(torch.zeros(h * w, dtype=torch.long))
        image_indices.append(torch.arange(length, length + h * w))
        positions.append(_grid_positions(pos, h, w, target_hw))
        segments.append((length, length + h * w, True))
        pos += max(h, w)
        length += h * w
    positions.append(_grid_positions(pos, target_hw[0], target_hw[1], target_hw))
    rope = _rope_table(torch.cat(positions), axes_dims).to(device)

    def _cat(parts):
        if not parts:
            return torch.zeros(0, dtype=torch.long, device=device)
        return torch.cat(parts).to(device=device, dtype=torch.long)

    return dict(
        encoder_seq_len=text_len,
        text_indices=_cat(text_indices),
        image_indices=_cat(image_indices),
        prefix_rope=rope[:length],
        target_rope=rope[length:],
        segments=tuple(segments),
    )


def prefix_cache_bytes(
    *, batch: int, prefix_len: int, num_layers: int, hidden: int, itemsize: int
) -> int:
    return 2 * num_layers * batch * prefix_len * hidden * itemsize


def choose_prefix_cache_store(
    option: str, *, need_bytes: int, free_gpu_bytes: int, free_host_bytes: int
) -> str | None:
    """Mirror ComfyUI ``select_prefix_cache``: "gpu", "cpu", or None to recompute.

    ``option`` is the QwenImage21Cache ``device`` widget (auto / gpu / cpu / off).
    """
    if option == "off":
        return None
    if option == "gpu":
        return "gpu" if free_gpu_bytes > 2 * need_bytes else None
    if option == "auto" and free_gpu_bytes > 4 * need_bytes:
        return "gpu"
    return "cpu" if free_host_bytes > 2 * need_bytes else None


class _HostKVRing:
    """Prefix K/V of one sample in pinned host memory, staged to the GPU per layer.

    Reading layer ``i`` prefetches layer ``i + 1`` on a side stream so the copy
    overlaps that block's compute; only two layers are resident at a time.
    """

    def __init__(self, num_layers: int):
        self.host: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        self.resident: dict[int, tuple[torch.Tensor, torch.Tensor, Any]] = {}
        self.stream = torch.cuda.Stream()

    def store(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        main = torch.cuda.current_stream()
        pair = tuple(
            torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in (key, value)
        )
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            for dst, src in zip(pair, (key, value)):
                dst.copy_(src, non_blocking=True)
                src.record_stream(self.stream)
        self.host[layer] = pair

    def _stage(self, layer: int, device) -> None:
        if (
            layer >= len(self.host)
            or layer in self.resident
            or self.host[layer] is None
        ):
            return
        # The host copy of this layer was queued on self.stream, so ordering holds.
        with torch.cuda.stream(self.stream):
            pair = tuple(t.to(device, non_blocking=True) for t in self.host[layer])
            event = torch.cuda.Event()
            event.record(self.stream)
        self.resident[layer] = (*pair, event)

    def fetch(self, layer: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        self._stage(layer, device)
        key, value, event = self.resident[layer]
        main = torch.cuda.current_stream()
        main.wait_event(event)
        key.record_stream(main)
        value.record_stream(main)
        for stale in [i for i in self.resident if i < layer]:
            del self.resident[stale]
        self._stage(layer + 1, device)
        return key, value


class HostPrefixKV(dict):
    """A per-layer prefix cache slot the DiT reads as ``cache["key"]`` / ``cache["value"]``.

    The dict holds CPU placeholders so ``if cache:`` stays truthy once filled.
    """

    def __init__(self, ring: _HostKVRing, layer: int, device):
        super().__init__()
        self._ring, self._layer, self._device = ring, layer, device

    def update(self, key=None, value=None, **kwargs):
        self._ring.store(self._layer, key, value)
        super().update(key=True, value=True)

    def __getitem__(self, name):
        key, value = self._ring.fetch(self._layer, self._device)
        return {"key": key, "value": value}[name]


def make_prefix_caches(store: str | None, *, batch: int, num_layers: int, device):
    if store is None:
        return None
    if store == "gpu":
        return [[{} for _ in range(num_layers)] for _ in range(batch)]
    caches = []
    for _ in range(batch):
        ring = _HostKVRing(num_layers)
        caches.append([HostPrefixKV(ring, i, device) for i in range(num_layers)])
    return caches


class QwenImage21ComfyUICond(msgspec.Struct, kw_only=True):
    """Worker-side conditioning for one ComfyUI cond batch in one sampler run."""

    context: Any
    layout: dict
    condition_latents: Any = None
    # None recomputes the prefix every step (QwenImage21Cache device=off, or no room).
    prefix_caches: Any = None
    last_sigma: float | None = None


class QwenImage21ComfyUIRunState(msgspec.Struct):
    conds: dict = {}


def _payload_ref_latents(payload: dict) -> list[torch.Tensor]:
    return [r for r in payload.get("ref_latents") or [] if r is not None]


def _free_memory(device) -> tuple[int, int]:
    import psutil

    free_gpu = 0
    if device.type == "cuda":
        # Blocks this process's caching allocator holds but does not use are free too.
        free_gpu = torch.cuda.mem_get_info(device)[0] + (
            torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        )
    return free_gpu, int(psutil.virtual_memory().available)


class QwenImage21ComfyUIStepStage(QwenImage21DenoisingStage):
    """ComfyUI owns the sampler loop; this runs the DiT once and returns the velocity."""

    def verify_input(self, batch, server_args) -> VerificationResult:
        return VerificationResult()

    def verify_output(self, batch, server_args) -> VerificationResult:
        return VerificationResult()

    def _prefix_caches_for(self, *, option: str, batch: int, prefix_len: int, device):
        arch = self.server_args.pipeline_config.dit_config.arch_config
        need = prefix_cache_bytes(
            batch=batch,
            prefix_len=prefix_len,
            num_layers=arch.num_layers,
            hidden=arch.hidden_size,
            itemsize=2,
        )
        free_gpu, free_host = _free_memory(device)
        store = choose_prefix_cache_store(
            option,
            need_bytes=need,
            free_gpu_bytes=free_gpu,
            free_host_bytes=free_host,
        )
        if store != "gpu":
            logger.info(
                "Qwen-Image 2.1 prefix K/V (%.2f GiB, %.2f GiB GPU free): %s",
                need / 2**30,
                free_gpu / 2**30,
                "pinned host memory" if store == "cpu" else "recomputed every step",
            )
        if store == "cpu" and device.type != "cuda":
            store = None
        return make_prefix_caches(
            store, batch=batch, num_layers=arch.num_layers, device=device
        )

    def _new_cond(self, *, batch, payload, latents) -> QwenImage21ComfyUICond:
        embeds = batch.prompt_embeds
        context = embeds[0] if isinstance(embeds, list) and embeds else None
        if context is None:
            raise RuntimeError(
                "Qwen-Image 2.1 ComfyUI step has no conditioning for this cond key; "
                "the worker lost its run state (restarted mid-run?)"
            )
        dit_config = self.server_args.pipeline_config.dit_config
        bsz, _, height, width = latents.shape
        device = latents.device
        dtype = torch.bfloat16
        context = context.to(device=device, dtype=dtype)
        if context.ndim == 2:
            context = context[None]
        if context.shape[0] != bsz:
            context = context.expand(bsz, -1, -1)
        refs = _payload_ref_latents(payload)
        layout = build_comfyui_layout(
            text_len=int(context.shape[1]),
            image_slots=[int(s) for s in payload.get("image_slots") or []],
            ref_shapes=[(int(r.shape[-2]), int(r.shape[-1])) for r in refs],
            target_hw=(height, width),
            axes_dims=dit_config.arch_config.axes_dims_rope,
            device=device,
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
            ).contiguous()
        return QwenImage21ComfyUICond(
            context=context,
            layout=layout,
            condition_latents=condition_latents,
            prefix_caches=self._prefix_caches_for(
                option=payload.get("prefix_cache", "auto"),
                batch=bsz,
                prefix_len=int(layout["prefix_rope"].shape[0]),
                device=device,
            ),
            last_sigma=payload.get("last_sigma"),
        )

    def _cond_for_step(self, batch, latents) -> QwenImage21ComfyUICond:
        extra = batch.extra or {}
        state = get_or_create_run_state(batch, QwenImage21ComfyUIRunState)
        bsz, _, height, width = latents.shape
        key = (extra.get("comfyui_cond_key") or "_", bsz, height, width)
        cond = state.conds.get(key)
        if cond is None:
            cond = self._new_cond(
                batch=batch, payload=extra.get(COND_EXTRA_KEY) or {}, latents=latents
            )
            state.conds[key] = cond
        return cond

    def forward(self, batch, server_args):
        begin_comfyui_run(batch)
        latents = batch.latents
        if latents is None or latents.ndim != 4:
            raise ValueError(
                "Qwen-Image 2.1 ComfyUI step expects latents shaped [B, C, H, W]"
            )
        cond = self._cond_for_step(batch, latents)
        bsz, _, height, width = latents.shape
        timestep = batch.timesteps.to(device=latents.device, dtype=torch.float32)
        timestep = timestep.reshape(-1).expand(bsz).contiguous()
        hidden_states = (
            latents.to(torch.bfloat16).flatten(2).transpose(1, 2).contiguous()
        )
        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ):
            output = self._predict_noise(
                current_model=self.transformer,
                latent_model_input=hidden_states,
                timestep=timestep,
                target_dtype=torch.bfloat16,
                guidance=None,
                encoder_hidden_states=cond.context,
                layouts=[cond.layout] * bsz,
                condition_latents=cond.condition_latents,
                prefix_caches=cond.prefix_caches,
            )
        batch.noise_pred = output.transpose(1, 2).reshape(
            bsz, output.shape[-1], height, width
        )
        if (
            cond.prefix_caches is not None
            and cond.last_sigma is not None
            and float(timestep[0]) / 1000.0 <= cond.last_sigma + 1e-6
        ):
            # Last sampler step for this cond: free its prefix K/V now instead of
            # holding it until the next run. A late extra call recomputes it.
            cond.prefix_caches = None
        return batch
