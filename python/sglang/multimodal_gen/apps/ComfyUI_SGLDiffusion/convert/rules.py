# SPDX-License-Identifier: Apache-2.0
"""What the SGLD workflow converter changes, as data.

Adding support for another community node usually means one table entry here.
"""

import json
import re

# Diffusion models SGLD integrated mode serves. LTX, Wan, detectors and other
# models stay on ComfyUI's stock loader.
_SGLD_NAME = re.compile(
    r"(?i)minimax|fasth3|(^|[^a-z0-9])h3|qwen|z[-_]?image|flux1|flux\.1|flux-1"
)
_NOT_SGLD = re.compile(
    r"(?i)ltx|wan2|hunyuan|rt_detr|sdpose|detr|yolo|sam3|flux2|flux\.2|krea"
)

# Stock loaders the converter swaps for SGLDUNETLoader (same widgets).
SGLD_LOADER = "SGLDUNETLoader"
STOCK_LOADERS = {"UNETLoader"}
SGLD_LOADERS = {"SGLDUNETLoader", "SGLDNativeModelLoader"}

# Same order as SGLDOptions' widgets (options.py _FIELDS); group headers are not serialized.
OPTIONS_WIDGETS = [
    "model_type", "num_gpus", "parallel_mode", "tp_size", "ulysses_degree",
    "ring_degree", "attention_backend", "attention_backend_config", "quantization",
    "int8_gemm", "cache_mode", "cache_dit_tuning", "cache_warmup_steps",
    "cache_threshold", "cache_max_cached_steps", "cache_fn_blocks", "execution_mode",
    "regional_compile", "memory_mode", "resident_layers", "prefetch_layers",
    "alignment_mode",
]  # fmt: skip
OPTIONS_DEFAULTS = {
    "model_type": "auto-detect", "num_gpus": 1, "parallel_mode": "auto", "tp_size": 1,
    "ulysses_degree": 1, "ring_degree": 1, "attention_backend": "auto",
    "attention_backend_config": "", "quantization": "none", "int8_gemm": "auto",
    "cache_mode": "off", "cache_dit_tuning": "auto", "cache_warmup_steps": 4,
    "cache_threshold": 0.24, "cache_max_cached_steps": 3, "cache_fn_blocks": 1,
    "execution_mode": "eager", "regional_compile": False, "memory_mode": "resident",
    "resident_layers": 0, "prefetch_layers": 0, "alignment_mode": "default",
}  # fmt: skip

SAGE = ("attention_backend", "sage_attention2")
SOL = ("attention_backend", "sol_attention")
CACHE = ("cache_mode", "cache_dit")
SPECTRUM = ("cache_mode", "spectrum")

# MODEL -> MODEL nodes that only change speed / memory of the DiT forward. On an
# SGLD model they are bypassed (the worker owns the forward); the value is the
# SGLDOptions setting that keeps the speed-up the author chose, or None.
SPEED_NODES = {
    # ComfyUI core
    "ModelAttentionBackend": None,  # see speed_setting(): sage only if it picks sage
    "BlockSparseAttention": SAGE,
    "EasyCache": CACHE,
    "LazyCache": CACHE,
    "TorchCompileModel": None,
    # KJNodes
    "PathchSageAttentionKJ": SAGE,
    "MiniMaxH3MemoryEfficientSageAttentionPatch": SAGE,
    "TorchCompileModelQwenImage": None,
    "TorchCompileModelAdvanced": None,
    "ModelPatchTorchSettings": None,
    "MiniMaxLowVRAMAttention": None,
    "MiniMaxChunkFeedForward": None,
    # MiniMax H3 community packs
    "SolAttnMiniMax": SOL,
    "MiniMaxH3ScheduledSolAttentionPatch": SOL,
    "MiniMaxH3MemoryEfficientSolAttentionPatch": SOL,
    "MiniMaxH3ChunkFeedForward": None,
    "MiniMaxH3FusedModulation": None,
    "H3SLAAttention": SAGE,
    "VedaSparseAttention": SAGE,
    "SpectrumApplyMiniMaxH3": SPECTRUM,
    "H3Cache": CACHE,
    "H3BlockCache": CACHE,
    "H3MiniMaxCache": CACHE,
    # Qwen-Image 2.1 T8 pack
    "QwenImage21SageAttentionT8": SAGE,
    "QwenImage21BlockCacheT8": CACHE,
    "QwenImage21SpectrumT8": SPECTRUM,
    # generic caches
    "TeaCache": CACHE,
    "MagCache": CACHE,
    "ApplyFBCacheOnModel": CACHE,
}

# Widget order of the nodes whose values the converter reads.
WIDGETS = {
    "ModelAttentionBackend": ["attention"],
    "SolAttnMiniMax": ["tau", "start_percent", "end_percent", "min_tokens", "sink_conditioning",
                       "morton", "morton_curve", "centroid_tail", "routed_cap_percent",
                       "reuse_qkv_memory", "verbose", "dense_blocks", "tau_profile"],
    "MiniMaxH3ScheduledSolAttentionPatch": ["enabled", "tau_start", "tau_end", "curve", "min_tokens",
                                            "strict", "dense_percent", "thresh_type", "int8_qk",
                                            "int8_pv", "sink_conditioning", "dense_blocks"],
    "MiniMaxH3MemoryEfficientSolAttentionPatch": ["enabled", "tau", "min_tokens", "strict",
                                                  "thresh_type", "int8_qk", "int8_pv",
                                                  "sink_conditioning", "dense_blocks"],
}  # fmt: skip

# Single-LoRA nodes: type -> (SGLD node, widget map from the old widgets to the new).
# SGLDLoraLoader widgets: lora_name, strength_model, nickname, target.
# SGLDLoraLoaderWithCLIP widgets: lora_name, strength_model, strength_clip (as LoraLoader).
LORA_NODES = {
    "LoraLoaderModelOnly": ("SGLDLoraLoader", lambda w: [w[0], w[1], "", "all"]),
    "LoraLoaderBypassModelOnly": ("SGLDLoraLoader", lambda w: [w[0], w[1], "", "all"]),
    "MiniMaxH3TurboLoRA": ("SGLDLoraLoader", lambda w: [w[0], w[1], "", "all"]),
    "LoraLoader": ("SGLDLoraLoaderWithCLIP", lambda w: [w[0], w[1], w[2]]),
    "LoraLoaderBypass": ("SGLDLoraLoaderWithCLIP", lambda w: [w[0], w[1], w[2]]),
}

# Multi-LoRA stack nodes: their model LoRAs move to a chain of SGLDLoraLoader
# nodes after them, and are switched off in the stack (its CLIP / trigger-word
# outputs keep working).
STACK_NODES = {
    "Power Lora Loader (rgthree)",
    "AUSBOSS_NODES_LoraLoader",
    "Lora Loader (LoraManager)",
}


def is_sgld_model(name) -> bool:
    if not isinstance(name, str):
        return False
    base = name.replace("\\", "/").split("/")[-1]
    return bool(_SGLD_NAME.search(base)) and not _NOT_SGLD.search(base)


def model_type(name, path=None):
    """SGLD model type of a diffusion-model file, from its name (and header if found)."""
    if not is_sgld_model(name):
        return None
    low = name.lower()
    if re.search(r"minimax|fasth3|(^|[^a-z0-9])h3", low):
        if "fasth3" in low or (path and _has_vsa_gate(path)):
            return "fast_h3"
        return "minimax_h3"
    if "qwen" in low:
        return "qwen_image21" if re.search(r"2[._-]?1", low) else "qwen_image"
    if re.search(r"z[-_]?image", low):
        return "zimage"
    return "flux"


def _has_vsa_gate(path) -> bool:
    try:
        with open(path, "rb") as f:
            size = int.from_bytes(f.read(8), "little")
            header = f.read(size).decode("utf-8", "ignore")
    except OSError:
        return False
    return "to_gate_compress" in header


def widget(node, name, order=None):
    order = order or WIDGETS.get(node.get("type"), [])
    values = node.get("widgets_values")
    if isinstance(values, dict):
        return values.get(name)
    if name in order and isinstance(values, list) and order.index(name) < len(values):
        return values[order.index(name)]
    return None


def speed_setting(node):
    """The SGLDOptions (key, value) a speed node maps to, or None."""
    kind = node.get("type")
    if kind == "ModelAttentionBackend":
        return SAGE if "sage" in str(widget(node, "attention")).lower() else None
    return SPEED_NODES.get(kind)


def sol_config(node) -> str:
    """The community Sol node's sparsity as SGLD sol_attn config.

    Those nodes run every step sparse at their own tau; SGLD's sol_attn defaults
    keep the first 10 steps and layers 0-1 dense, which leaves a turbo (8-12
    step) sampler almost dense.
    """
    config = {"dense_steps": 0, "dense_layers": ""}
    tau = widget(node, "tau")
    if tau is None:
        tau = widget(node, "tau_start")
    if isinstance(tau, (int, float)) and not isinstance(tau, bool):
        config["tau"] = float(tau)
    thresh = widget(node, "thresh_type")
    if isinstance(thresh, str):
        config["thresh_type"] = thresh
    dense = widget(node, "dense_blocks")
    # SGLD takes non-negative indices and ranges only ("0-3,47").
    if (
        isinstance(dense, str)
        and re.fullmatch(r"[\d,\s-]*", dense)
        and not re.search(r"(^|,)\s*-", dense)
    ):
        config["dense_layers"] = dense
    return json.dumps(config, sort_keys=True)


def stack_loras(node):
    """[(lora_name, strength)] switched on in a LoRA stack node; switches them off."""
    kind, values = node.get("type"), node.get("widgets_values")
    found = []
    items = values if isinstance(values, list) else []
    if kind == "Power Lora Loader (rgthree)":
        for v in items:
            if (
                isinstance(v, dict)
                and v.get("lora") not in (None, "", "None")
                and "on" in v
            ):
                if v["on"]:
                    found.append((v["lora"], float(v.get("strength", 1.0))))
                v["on"] = False
    elif kind == "AUSBOSS_NODES_LoraLoader":
        for i, v in enumerate(items):
            if isinstance(v, str) and v.lstrip().startswith("["):
                try:
                    loras = json.loads(v)
                except ValueError:
                    continue
                for it in loras:
                    if isinstance(it, dict) and it.get("enabled"):
                        found.append((it["name"], float(it.get("strength", 1.0))))
                        it["enabled"] = False
                items[i] = json.dumps(loras)
    elif kind == "Lora Loader (LoraManager)":
        for v in items:
            loras = v.get("__value__", v) if isinstance(v, dict) else v
            if not isinstance(loras, list):
                continue
            for it in loras:
                if isinstance(it, dict) and "name" in it and it.get("active", True):
                    name = it["name"]
                    if not name.endswith(".safetensors"):
                        name += ".safetensors"
                    found.append((name, float(it.get("strength", 1.0))))
                    it["active"] = False
    return found
