# SPDX-License-Identifier: Apache-2.0
"""ComfyUI checkpoint quantization for diffusion models."""

from __future__ import annotations

import fnmatch
import logging
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant_kernel_adapter import (
    ComfyQuantKernelBackend,
    apply_int8_linear,
    select_int8_kernel_backend,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_quant_scanner import (
    ComfyLayerQuantDescription,
    checkpoint_has_comfy_quant_layers,
    scan_comfy_quant_layers,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

logger = logging.getLogger(__name__)

_INT8_FORMATS = frozenset({"int8_tensorwise"})
_FP8_FORMATS = frozenset({"float8_e4m3fn", "float8_e5m2"})
_NVFP4_FORMATS = frozenset({"nvfp4"})


def _use_comfy_nvfp4_dequant_path(desc: ComfyLayerQuantDescription) -> bool:
    """Use bf16 matmul after load-time dequant (Comfy full_precision / pre-SM100)."""

    if desc.full_precision_matrix_mult:
        return True
    capability = current_platform.get_device_capability()
    if capability is not None and capability.to_int() < 100:
        return True
    return False


class ComfyQuantConfig(QuantizationConfig):
    """Per-layer ComfyUI quantization inferred from ``.comfy_quant`` blobs."""

    def __init__(
        self,
        layers: dict[str, ComfyLayerQuantDescription] | None = None,
        ignore: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.layers = layers or {}
        self.ignore = ignore or []

    @classmethod
    def get_name(cls) -> str:
        return "comfy_quant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ComfyQuantConfig":
        raw_layers = config.get("layers", {})
        layers = {
            prefix: (
                desc
                if isinstance(desc, ComfyLayerQuantDescription)
                else ComfyLayerQuantDescription.from_json(prefix, desc)
            )
            for prefix, desc in raw_layers.items()
        }
        return cls(layers=layers, ignore=list(config.get("ignore", [])))

    @classmethod
    def from_safetensors_list(cls, file_paths: list[str]) -> "ComfyQuantConfig | None":
        if not checkpoint_has_comfy_quant_layers(file_paths):
            return None
        return cls(layers=scan_comfy_quant_layers(file_paths))

    def _is_layer_ignored(self, prefix: str) -> bool:
        for pattern in self.ignore:
            if fnmatch.fnmatch(prefix, pattern):
                return True
        return False

    def _layer_description(self, prefix: str) -> ComfyLayerQuantDescription | None:
        if self._is_layer_ignored(prefix):
            return None
        return self.layers.get(prefix)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None

        desc = self._layer_description(prefix)
        if desc is None:
            return UnquantizedLinearMethod()

        if desc.format in _INT8_FORMATS:
            return ComfyInt8ConvRotLinearMethod(self, desc)
        if desc.format in _FP8_FORMATS:
            return Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
        if desc.format in _NVFP4_FORMATS:
            if _use_comfy_nvfp4_dequant_path(desc):
                return UnquantizedLinearMethod()
            return ModelOptFp4LinearMethod(
                ModelOptFp4Config(
                    is_checkpoint_nvfp4_serialized=True,
                    group_size=16,
                    checkpoint_weight_scale_layout="swizzled",
                    swap_weight_nibbles=True,
                )
            )

        logger.warning(
            "Unsupported comfy_quant format %r for %s; using unquantized linear",
            desc.format,
            prefix,
        )
        return UnquantizedLinearMethod()


class ComfyInt8ConvRotLinearMethod(LinearMethodBase):
    """INT8 tensorwise (+ optional ConvRot) linear from Comfy checkpoints."""

    def __init__(
        self,
        quant_config: ComfyQuantConfig,
        layer_desc: ComfyLayerQuantDescription,
    ) -> None:
        self.quant_config = quant_config
        self.layer_desc = layer_desc
        self.kernel_backend = select_int8_kernel_backend(
            force_full_precision=layer_desc.full_precision_matrix_mult,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.comfy_quant_convrot = self.layer_desc.convrot
        layer.comfy_quant_convrot_groupsize = self.layer_desc.convrot_groupsize
        layer.comfy_quant_kernel_backend = self.kernel_backend

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        scale = ChannelQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "comfy_quant_kernel_backend", None) == (
            ComfyQuantKernelBackend.SGL_KERNEL
        ):
            # sgl_kernel int8_scaled_mm expects column-major weight layout.
            layer.weight = torch.nn.Parameter(layer.weight.data.t(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        backend = getattr(
            layer,
            "comfy_quant_kernel_backend",
            ComfyQuantKernelBackend.TORCH,
        )
        return apply_int8_linear(
            backend=backend,
            x=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            bias=bias,
            convrot=bool(getattr(layer, "comfy_quant_convrot", False)),
            convrot_groupsize=int(
                getattr(layer, "comfy_quant_convrot_groupsize", 256)
            ),
        )


__all__ = [
    "ComfyInt8ConvRotLinearMethod",
    "ComfyQuantConfig",
]
