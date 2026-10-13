"""Make room on the GPU for the SGLD worker.

The worker is a separate process, so ComfyUI's memory manager never counts it:
a text encoder left on the GPU after encoding stays there while the worker loads
the DiT or allocates activations. ComfyUI's own free_memory() measures device-wide
free memory (other processes included), so asking it for room is enough.
"""

import logging
import os

logger = logging.getLogger(__name__)

_GIB = 1024**3
# Activation headroom the worker needs during sampling, as a fraction of the card.
_SAMPLING_HEADROOM_FRACTION = 0.25


def free_comfy_vram(memory_required: int | None, keep_patcher=None) -> None:
    """Unload ComfyUI's own models (text encoder, VAE) from the worker's GPU
    until `memory_required` bytes are free; None means the sampling headroom.
    keep_patcher stays loaded."""
    try:
        from comfy import model_management
    except ImportError:
        return
    device = model_management.get_torch_device()
    if device.type != "cuda":
        return
    if memory_required is None:
        memory_required = int(
            model_management.get_total_memory(device) * _SAMPLING_HEADROOM_FRACTION
        )
    keep = [
        loaded
        for loaded in model_management.current_loaded_models
        if keep_patcher is not None and loaded.model is keep_patcher
    ]
    unloaded = model_management.free_memory(memory_required, device, keep_loaded=keep)
    if unloaded:
        names = ", ".join(type(m.model.model).__name__ for m in unloaded)
        logger.info(
            "SGLD: moved %s off %s to make room for the worker (%.1f GiB requested)",
            names,
            device,
            memory_required / _GIB,
        )


def worker_weight_bytes(model_path: str, sgld_options: dict) -> int:
    """GPU bytes the worker's DiT weights need on one rank, plus a 2 GiB margin."""
    if sgld_options.get("dit_layerwise_offload"):
        return 4 * _GIB
    path = sgld_options.get("transformer_weights_path") or model_path
    if os.path.isdir(path):
        size = sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, files in os.walk(path)
            for name in files
            if name.endswith((".safetensors", ".gguf"))
        )
        if sgld_options.get("quantization") == "convrot_int8":
            size //= 2  # BF16 directory quantized online to INT8
    else:
        size = os.path.getsize(path)
    tp = int(sgld_options.get("tp_size") or 1)
    return size // max(tp, 1) + 2 * _GIB
