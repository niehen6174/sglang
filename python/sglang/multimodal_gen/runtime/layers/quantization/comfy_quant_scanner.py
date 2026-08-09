# SPDX-License-Identifier: Apache-2.0
"""Scan ComfyUI ``.comfy_quant`` tensor keys from safetensors checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from safetensors import safe_open

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

COMFY_QUANT_SUFFIX = ".comfy_quant"


@dataclass(frozen=True)
class ComfyLayerQuantDescription:
    """Per-module quantization description from a Comfy ``.comfy_quant`` blob."""

    prefix: str
    format: str
    convrot: bool = False
    convrot_groupsize: int = 256
    linear_dtype: str | None = None
    full_precision_matrix_mult: bool = False
    num_experts: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, prefix: str, payload: dict[str, Any]) -> "ComfyLayerQuantDescription":
        fmt = str(payload.get("format", "")).strip()
        if not fmt:
            raise ValueError(f"Missing format in comfy_quant blob for {prefix!r}")
        return cls(
            prefix=prefix,
            format=fmt,
            convrot=bool(payload.get("convrot", False)),
            convrot_groupsize=int(payload.get("convrot_groupsize", 256)),
            linear_dtype=payload.get("linear_dtype"),
            full_precision_matrix_mult=bool(
                payload.get("full_precision_matrix_mult", False)
            ),
            num_experts=(
                int(payload["num_experts"])
                if payload.get("num_experts") is not None
                else None
            ),
            raw=dict(payload),
        )


def _decode_comfy_quant_blob(blob: Any) -> dict[str, Any]:
    if hasattr(blob, "tolist"):
        raw = bytes(blob.tolist()).decode("utf-8")
    elif isinstance(blob, (bytes, bytearray)):
        raw = bytes(blob).decode("utf-8")
    else:
        raw = str(blob)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("comfy_quant blob must decode to a JSON object")
    return payload


def scan_comfy_quant_layers(file_paths: list[str]) -> dict[str, ComfyLayerQuantDescription]:
    """Scan safetensors keys ending in ``.comfy_quant``.

    Comfy checkpoints store per-layer quantization metadata as UTF-8 JSON encoded
    into uint8 tensors. Safetensors ``__metadata__`` is often empty for these
    files, so key scanning is the reliable source of truth.
    """
    layers: dict[str, ComfyLayerQuantDescription] = {}
    for file_path in file_paths:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.endswith(COMFY_QUANT_SUFFIX):
                    continue
                prefix = key[: -len(COMFY_QUANT_SUFFIX)]
                payload = _decode_comfy_quant_blob(handle.get_tensor(key))
                desc = ComfyLayerQuantDescription.from_json(prefix, payload)
                existing = layers.get(prefix)
                if existing is not None and existing.raw != desc.raw:
                    raise ValueError(
                        f"Conflicting comfy_quant metadata for {prefix!r} across shards"
                    )
                layers[prefix] = desc

    if layers:
        logger.info(
            "Scanned %d comfy_quant layer descriptions from %d safetensors file(s)",
            len(layers),
            len(file_paths),
        )
    return layers


def checkpoint_has_comfy_quant_layers(file_paths: list[str]) -> bool:
    for file_path in file_paths:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            if any(key.endswith(COMFY_QUANT_SUFFIX) for key in handle.keys()):
                return True
    return False


__all__ = [
    "COMFY_QUANT_SUFFIX",
    "ComfyLayerQuantDescription",
    "checkpoint_has_comfy_quant_layers",
    "scan_comfy_quant_layers",
]
