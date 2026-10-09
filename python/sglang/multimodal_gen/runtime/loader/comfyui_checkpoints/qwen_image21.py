# SPDX-License-Identifier: Apache-2.0
"""ComfyUI Qwen-Image 2.1 checkpoint spec.

Comfy-Org's single-file DiT (``qwen_image_2.1_bf16.safetensors``) keeps the
native parameter names except for the SwiGLU input projection, which ComfyUI
fuses row-wise as ``img_mlp.gate_up = [gate; up]``. SGLang keeps the two halves
as ``gate_layer`` and ``proj``.
"""

import re
from typing import Any

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import (
    QwenImage21ArchConfig,
    QwenImage21DitConfig,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    WeightIterator,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_PREFIX_RE = re.compile(r"^(?:model\.diffusion_model\.|diffusion_model\.)")
# Fused SwiGLU input projection; INT8 ConvRot files add a per-row weight_scale.
_GATE_UP_RE = re.compile(r"^(.*\.img_mlp)\.gate_up\.(weight|weight_scale)$")
_GATE_UP_PREFIX = ".img_mlp.gate_up"
_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")


def strip_comfyui_prefix(name: str) -> str:
    return _PREFIX_RE.sub("", name)


def arch_from_checkpoint_shapes(shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """Mirror ComfyUI's ``detect_unet_config`` for ``qwen_image21``."""
    shapes = {strip_comfyui_prefix(k): tuple(v) for k, v in shapes.items()}
    img_in = shapes["img_in.weight"]
    head_dim = shapes["transformer_blocks.0.attn.norm_q.weight"][0]
    inner_dim = img_in[0]
    gate_up = shapes.get("transformer_blocks.0.img_mlp.gate_up.weight")
    if gate_up is not None:
        mlp_ratio = gate_up[0] // 2 // inner_dim
    else:
        mlp_ratio = shapes["transformer_blocks.0.img_mlp.proj.weight"][0] // inner_dim
    num_layers = 1 + max(
        int(m.group(1)) for m in map(_BLOCK_RE.match, shapes) if m is not None
    )
    return dict(
        in_channels=img_in[1],
        out_channels=shapes["proj_out.weight"][0],
        num_layers=num_layers,
        attention_head_dim=head_dim,
        num_attention_heads=inner_dim // head_dim,
        context_in_dim=shapes["txt_in.text_norm.weight"][0],
        mlp_ratio=mlp_ratio,
    )


def _read_shapes(model_path: str) -> dict[str, tuple[int, ...]] | None:
    if not model_path or not model_path.endswith(".safetensors"):
        return None
    try:
        from safetensors import safe_open

        with safe_open(model_path, framework="pt", device="cpu") as handle:
            return {
                key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()
            }
    except Exception:
        logger.warning("Could not read %s header", model_path, exc_info=True)
        return None


def _build_dit_config(server_args: ServerArgs) -> QwenImage21DitConfig:
    arch_kwargs: dict[str, Any] = {}
    shapes = _read_shapes(server_args.model_path)
    if shapes:
        try:
            arch_kwargs = arch_from_checkpoint_shapes(shapes)
        except KeyError:
            logger.warning(
                "Unrecognized Qwen-Image 2.1 checkpoint layout; using default arch"
            )
    dit_config = QwenImage21DitConfig(arch_config=QwenImage21ArchConfig(**arch_kwargs))
    server_args.pipeline_config.dit_config = dit_config
    return dit_config


def _convert_weights(weights: WeightIterator, dit_config: Any) -> WeightIterator:
    """Split ComfyUI's fused ``img_mlp.gate_up`` into ``gate_layer`` / ``proj``."""
    arch = dit_config.arch_config
    ffn_dim = arch.hidden_size * arch.mlp_ratio
    for name, tensor in weights:
        name = strip_comfyui_prefix(name)
        if name.endswith(".comfy_quant"):
            # Read through quant_markers; the custom iterator bypasses the key filter.
            continue
        match = _GATE_UP_RE.match(name)
        if match is None:
            yield name, tensor
            continue
        if tensor.shape[0] != 2 * ffn_dim:
            raise ValueError(
                f"{name} must have {2 * ffn_dim} rows, got {tuple(tensor.shape)}"
            )
        mlp, param = match.groups()
        # Row split is exact for INT8 too: ConvRot rotates the input (column) axis
        # and the scale is per output row.
        yield f"{mlp}.gate_layer.{param}", tensor[:ffn_dim]
        yield f"{mlp}.proj.{param}", tensor[ffn_dim:]


def _quant_markers(safetensors_list: list[str]) -> dict[str, dict[str, Any]]:
    from sglang.multimodal_gen.runtime.utils.quantization_utils import (
        inspect_comfy_quant_markers,
    )

    markers = inspect_comfy_quant_markers(
        safetensors_list, param_name_mapper=strip_comfyui_prefix
    )
    return split_gate_up_markers(markers)


def split_gate_up_markers(
    markers: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out = {}
    for prefix, marker in markers.items():
        if prefix.endswith(_GATE_UP_PREFIX):
            mlp = prefix[: -len(".gate_up")]
            out[f"{mlp}.gate_layer"] = dict(marker)
            out[f"{mlp}.proj"] = dict(marker)
        else:
            out[prefix] = marker
    return out


register_comfyui_checkpoint(
    "QwenImage21Pipeline",
    ComfyUICheckpointSpec(
        dit_cls_name="QwenImage21Transformer2DModel",
        build_dit_config=_build_dit_config,
        convert_weights=_convert_weights,
        quant_markers=_quant_markers,
    ),
)
