# SPDX-License-Identifier: Apache-2.0
"""Native, TP-foldable Qwen3-VL layer-50 encoder for MiniMax H3."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
    MiniMaxH3Qwen3VLConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant import (
    ComfyQuantConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant_kernel_adapter import (
    ensure_comfy_kitchen_nvfp4_lut_materialized,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant_scanner import (
    ComfyLayerQuantDescription,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLModel

MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120
_LAYER_WEIGHT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def remap_comfy_h3_te_weight_name(name: str) -> str:
    """Map Comfy single-file TE keys to native Qwen3-VL parameter names."""

    if name.startswith("model.layers."):
        return "model.language_model." + name[len("model.") :]
    if name.startswith("model.embed_tokens."):
        suffix = name[len("model.embed_tokens.") :]
        return f"model.language_model.embed_tokens.{suffix}"
    if name.startswith("visual."):
        return "model.visual." + name[len("visual.") :]
    return name


def remap_comfy_h3_te_quant_prefix(prefix: str) -> str:
    """Map Comfy quant layer prefixes to Qwen3VLTextModel linear prefixes."""

    if prefix.startswith("model.layers."):
        return prefix[len("model.") :]
    if prefix == "model.embed_tokens":
        return "embed_tokens"
    return prefix


def remap_comfy_h3_te_quant_config(
    quant_config: ComfyQuantConfig,
) -> ComfyQuantConfig:
    remapped: dict[str, ComfyLayerQuantDescription] = {}
    for prefix, desc in quant_config.layers.items():
        new_prefix = remap_comfy_h3_te_quant_prefix(prefix)
        remapped[new_prefix] = ComfyLayerQuantDescription(
            prefix=new_prefix,
            format=desc.format,
            convrot=desc.convrot,
            convrot_groupsize=desc.convrot_groupsize,
            linear_dtype=desc.linear_dtype,
            full_precision_matrix_mult=desc.full_precision_matrix_mult,
            num_experts=desc.num_experts,
            raw=dict(desc.raw),
        )
    return ComfyQuantConfig(layers=remapped, ignore=list(quant_config.ignore))


def apply_quant_weight_postprocess(model: nn.Module) -> None:
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and hasattr(
            quant_method, "process_weights_after_loading"
        ):
            quant_method.process_weights_after_loading(module)


def _attach_comfy_h3_te_pre_quant_scales(
    model: nn.Module,
    pending: dict[str, torch.Tensor],
) -> None:
    """Register Comfy NVFP4 ``pre_quant_scale`` buffers on TE linears."""

    for name, scale in pending.items():
        if not name.endswith(".pre_quant_scale"):
            continue
        module_name = name[: -len(".pre_quant_scale")]
        module = model
        for part in module_name.split("."):
            module = getattr(module, part)
        module.register_buffer(
            "pre_quant_scale",
            scale.detach().to(dtype=torch.bfloat16),
            persistent=False,
        )


def _is_unconsumed_checkpoint_weight(name: str) -> bool:
    """Weights intentionally absent from the layer-50 feature extractor."""

    if name == "lm_head.weight" or name.startswith("model.language_model.norm."):
        return True
    match = _LAYER_WEIGHT_RE.match(name)
    return bool(match and int(match.group(1)) >= MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER)


def _is_int8_embedding_checkpoint_weight(name: str, loaded_weight: torch.Tensor) -> bool:
    """Comfy H3 TE stores ``embed_tokens.weight`` as int8 with per-row ``weight_scale``."""

    return (
        name.endswith(".embed_tokens.weight")
        and loaded_weight.dtype in (torch.int8, torch.uint8)
    )


class MiniMaxH3Qwen3VLEncoder(TextEncoder):
    """Qwen3-VL-32B multimodal backbone ending at hidden_states[50].

    The component loader builds and loads this module under the encoder-folding
    TP group. A TP=1/SP=8 DiT deployment therefore shards the encoder over all
    eight otherwise-idle ranks during encoding.
    """

    supports_dp_encode = True

    @staticmethod
    def should_materialize_checkpoint_weight(name: str) -> bool:
        return (
            "rotary_emb.inv_freq" not in name
            and not _is_unconsumed_checkpoint_weight(name)
        )

    def __init__(
        self,
        config: MiniMaxH3Qwen3VLConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config)
        arch = config.arch_config
        selected_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        if int(arch.text_config.num_hidden_layers) != selected_layer:
            raise ValueError(
                "MiniMax H3 Qwen3-VL config must be trimmed to "
                f"{selected_layer} language layers before construction"
            )
        self.model = Qwen3VLModel(
            arch,
            use_tensor_parallel=True,
            quant_config=quant_config,
        )
        # H3 consumes the unnormalized output immediately after layer 49.
        self.model.language_model.norm = nn.Identity()
        self.image_token_id = int(arch.image_token_id)
        self.video_token_id = int(arch.video_token_id)
        self.selected_lm_layer = selected_layer
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _resolve_compute_device(self) -> torch.device:
        param_devices = {param.device.type for param in self.parameters()}
        if "cpu" in param_devices:
            return torch.device("cpu")
        if len(param_devices) == 1:
            return next(iter({param.device for param in self.parameters()}))
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        return BaseEncoderOutput(last_hidden_state=outputs.last_hidden_state)

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )

        host_ids = input_ids.to(device="cpu", dtype=torch.long)[None]
        host_image_grid_thw = (
            image_grid_thw.to(device="cpu", dtype=torch.long)
            if image_grid_thw is not None
            else None
        )
        host_video_grid_thw = (
            video_grid_thw.to(device="cpu", dtype=torch.long)
            if video_grid_thw is not None
            else None
        )
        position_ids = None
        if host_image_grid_thw is not None or host_video_grid_thw is not None:
            position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid_thw,
                host_video_grid_thw,
                attention_mask=torch.ones_like(host_ids),
            )
        force_cpu = getattr(self, "_force_cpu_forward", False) or getattr(
            getattr(self, "module", None), "_force_cpu_forward", False
        )
        if force_cpu:
            compute_device = torch.device("cpu")
        else:
            compute_device = self._resolve_compute_device()
        if compute_device.type == "cpu":
            self.model.to("cpu")
        ids = host_ids.to(compute_device)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
            "use_cache": False,
        }
        if position_ids is not None:
            call_kwargs["position_ids"] = position_ids.to(compute_device)
        if pixel_values is not None:
            call_kwargs["pixel_values"] = pixel_values.to(
                compute_device, torch.bfloat16
            )
            call_kwargs["image_grid_thw"] = host_image_grid_thw
        if pixel_values_videos is not None:
            call_kwargs["pixel_values_videos"] = pixel_values_videos.to(
                compute_device, torch.bfloat16
            )
            call_kwargs["video_grid_thw"] = host_video_grid_thw

        with torch.device(compute_device):
            if force_cpu and hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
            hidden = self.model(**call_kwargs).last_hidden_state[0].to(torch.bfloat16)
        expected_shape = [int(ids.shape[1]), self.hidden_dim]
        if list(hidden.shape) != expected_shape:
            raise ValueError(
                f"unexpected hidden shape {list(hidden.shape)}, "
                f"expected {expected_shape}"
            )
        return hidden

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params = dict(self.named_parameters(remove_duplicate=False))
        pending: dict[str, torch.Tensor] = {}
        for name, loaded_weight in weights:
            if name.endswith(".comfy_quant"):
                continue
            mapped = remap_comfy_h3_te_weight_name(name)
            if loaded_weight.dtype in (torch.uint8, torch.int8):
                pending[mapped] = torch.from_numpy(loaded_weight.numpy().copy())
            elif loaded_weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                pending[mapped] = (
                    torch.from_numpy(loaded_weight.view(torch.uint8).numpy().copy())
                    .view(loaded_weight.dtype)
                )
            else:
                pending[mapped] = loaded_weight.detach().clone().to(device="cpu")

        loaded: set[str] = set()
        aux_suffixes = (
            ".weight_scale",
            ".weight_scale_2",
            ".pre_quant_scale",
            ".input_scale",
        )
        for name, loaded_weight in pending.items():
            if not self.should_materialize_checkpoint_weight(name):
                continue
            param = params.get(name)
            if param is None:
                if any(name.endswith(suffix) for suffix in aux_suffixes):
                    continue
                raise KeyError(
                    f"Unexpected MiniMax H3 Qwen3-VL checkpoint weight: {name}"
                )

            if (
                name.endswith(".weight")
                and loaded_weight.dtype in (torch.uint8, torch.int8)
                and param.dtype in (torch.bfloat16, torch.float16, torch.float32)
                and loaded_weight.ndim == 2
                and loaded_weight.shape[1] * 2 == param.shape[1]
            ):
                scale_2_name = name.replace(".weight", ".weight_scale_2")
                block_scale_name = name.replace(".weight", ".weight_scale")
                per_tensor_scale = pending.get(scale_2_name)
                block_scales = pending.get(block_scale_name)
                if per_tensor_scale is None or block_scales is None:
                    raise KeyError(
                        f"Missing NVFP4 scales for {name!r}: "
                        f"{scale_2_name!r}, {block_scale_name!r}"
                    )
                ensure_comfy_kitchen_nvfp4_lut_materialized()
                from comfy_kitchen.backends.eager.quantization import (
                    dequantize_nvfp4 as eager_dequantize_nvfp4,
                )

                loaded_weight = eager_dequantize_nvfp4(
                    loaded_weight.cpu(),
                    per_tensor_scale.cpu(),
                    block_scales.cpu(),
                    torch.bfloat16,
                )

            if _is_int8_embedding_checkpoint_weight(name, loaded_weight):
                scale_name = name.replace(".weight", ".weight_scale")
                scale = pending.get(scale_name)
                if scale is None:
                    raise KeyError(
                        f"Missing int8 embedding scale for {name!r} ({scale_name!r})"
                    )
                loaded_weight = loaded_weight.to(torch.float32) * scale.to(torch.float32)
                loaded_weight = loaded_weight.to(param.dtype)

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                if param.device.type == "meta":
                    param.data = loaded_weight.to(dtype=param.dtype, device="cpu")
                elif getattr(param, "weight_loader", None) is default_weight_loader:
                    weight_loader(param, loaded_weight.to(param.dtype))
                else:
                    weight_loader(param, loaded_weight)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load MiniMax H3 Qwen3-VL weight "
                    f"{name!r}: checkpoint={tuple(loaded_weight.shape)}, "
                    f"parameter={tuple(param.shape)}"
                ) from exc
            loaded.add(name)
        _attach_comfy_h3_te_pre_quant_scales(self, pending)
        apply_quant_weight_postprocess(self)
        return loaded


EntryClass = MiniMaxH3Qwen3VLEncoder

__all__ = [
    "MiniMaxH3Qwen3VLEncoder",
    "_attach_comfy_h3_te_pre_quant_scales",
    "apply_quant_weight_postprocess",
    "remap_comfy_h3_te_quant_config",
    "remap_comfy_h3_te_weight_name",
]
