"""Replace speed-only model patch nodes that sit on an SGLD model.

The DiT runs in the SGLD worker, which picks its own attention kernel and step
caching (SGLDOptions), so attention / sparse-attention / cache / compile patch
nodes cannot change anything there. Many of them also refuse a model that is not
ComfyUI's native class. Before the prompt is queued, each such node whose MODEL
input comes from an SGLD loader is removed and its consumers are wired to its
input. The speed-up it asked for (Sage / sparse attention, step caching) moves to
the loader's SGLDOptions unless the user chose a value there; without that the
worker would run dense and uncached, slower than the stock workflow. Nodes that change what the model computes (ControlNet, block patches)
are left in place, so they still fail loudly at sampling time.
"""

import functools
import json
import logging
import re

logger = logging.getLogger(__name__)

SGLD_LOADERS = {"SGLDUNETLoader", "SGLDNativeModelLoader"}

# MODEL -> MODEL nodes that only change speed / memory of the DiT forward, and
# the SGLDOptions setting that keeps the speed-up the workflow author chose
# (None: nothing to carry over, e.g. compile or ComfyUI VRAM tricks).
_SAGE = ("attention_backend", "sage_attention2")
_SOL = ("attention_backend", "sol_attention")
_CACHE = ("cache_mode", "cache_dit")
_SPECTRUM = ("cache_mode", "spectrum")
SPEED_ONLY_PATCHES = {
    # ComfyUI core
    "ModelAttentionBackend": None,  # resolved from its widget, see _sgld_setting
    "BlockSparseAttention": _SAGE,
    "EasyCache": _CACHE,
    "LazyCache": _CACHE,
    "TorchCompileModel": None,
    # KJNodes
    "PathchSageAttentionKJ": _SAGE,
    "MiniMaxH3MemoryEfficientSageAttentionPatch": _SAGE,
    "TorchCompileModelQwenImage": None,
    "TorchCompileModelAdvanced": None,
    "ModelPatchTorchSettings": None,
    "MiniMaxLowVRAMAttention": None,
    "MiniMaxChunkFeedForward": None,
    # MiniMax H3 community
    "SolAttnMiniMax": _SOL,
    "MiniMaxH3ScheduledSolAttentionPatch": _SOL,
    "MiniMaxH3MemoryEfficientSolAttentionPatch": _SOL,
    "H3SLAAttention": _SAGE,
    "VedaSparseAttention": _SAGE,
    "SpectrumApplyMiniMaxH3": _SPECTRUM,
    "H3Cache": _CACHE,
    "H3BlockCache": _CACHE,
    "H3MiniMaxCache": _CACHE,
    # Qwen-Image 2.1 T8 pack
    "QwenImage21SageAttentionT8": _SAGE,
    "QwenImage21BlockCacheT8": _CACHE,
    "QwenImage21SpectrumT8": _SPECTRUM,
    # generic caches
    "TeaCache": _CACHE,
    "MagCache": _CACHE,
    "ApplyFBCacheOnModel": _CACHE,
}
# SGLDOptions values that mean "not chosen"; anything else the user set wins.
_UNSET = {
    "attention_backend": "auto",
    "attention_backend_config": "",
    "cache_mode": "off",
}


# MODEL-only LoRA nodes that patch DiT internals themselves (so they bypass
# comfy.sd.load_lora_for_models): class -> (lora file input, strength input).
CUSTOM_LORA_NODES = {
    "MiniMaxH3TurboLoRA": ("lora_name", "strength"),
    "LoraLoaderBypassModelOnly": ("lora_name", "strength_model"),
}


def _sgld_loader(prompt: dict, node_id: str, seen: set) -> str | None:
    """The SGLD loader the MODEL reaching ``node_id`` comes from, if any.

    Follows ``model`` first, then any other link, so switch / reroute nodes
    (``on_true`` / ``on_false``, ``input``) between the loader and the patch count.
    """
    if node_id not in prompt or node_id in seen:
        return None
    seen.add(node_id)
    node = prompt[node_id]
    if node.get("class_type") in SGLD_LOADERS:
        return node_id
    inputs = node.get("inputs", {})
    links = sorted(
        (key != "model", str(value[0]))
        for key, value in inputs.items()
        if isinstance(value, list) and len(value) == 2
    )
    for _, source in links:
        found = _sgld_loader(prompt, source, seen)
        if found:
            return found
    return None


def _sgld_setting(node: dict):
    setting = SPEED_ONLY_PATCHES[node["class_type"]]
    if node["class_type"] == "ModelAttentionBackend":
        widget = str(node.get("inputs", {}).get("attention", "")).lower()
        setting = _SAGE if "sage" in widget else None
    if setting == _SOL and not _available(_SOL[1]):
        setting = _SAGE
    if setting == _SAGE and not _available(_SAGE[1]):
        setting = None
    return setting


def _available(backend: str) -> bool:
    from ..options import _ATTENTION_IDS
    from .preflight import check_attention_backends

    try:
        check_attention_backends(
            {"attention_backend": _ATTENTION_IDS.get(backend, backend)}
        )
    except ValueError:
        return False
    return True


def _model_type(loader: dict) -> str | None:
    """The SGLD model type of an SGLDUNETLoader file (header-only detection)."""
    name = loader.get("inputs", {}).get("unet_name")
    if loader.get("class_type") != "SGLDUNETLoader" or not isinstance(name, str):
        return None
    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", name)
    return _detect(path) if path else None


@functools.lru_cache(maxsize=64)
def _detect(path: str) -> str | None:
    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.minimax_h3 import (
        has_vsa_gate,
    )

    from .generator import SGLDiffusionGenerator

    try:
        model_type = SGLDiffusionGenerator.shared().get_comfyui_model(path)[2]
    except Exception:  # unreadable or unknown file: the loader reports it
        return None
    return (
        "fast_h3" if model_type == "minimax_h3" and has_vsa_gate(path) else model_type
    )


def _supported(model_type: str | None, settings: dict) -> dict:
    """Drop the settings the model cannot run (preflight's model rules)."""
    from .preflight import check_model_options

    if model_type is None:
        return {}
    if model_type == "fast_h3":
        # FastH3 runs the VSA-H3 sparse attention it was distilled with by default.
        settings = {k: v for k, v in settings.items() if k != "attention_backend"}
    kept = {}
    for key, value in settings.items():
        option = {key: _ATTENTION_ID(value) if key == "attention_backend" else value}
        try:
            check_model_options(model_type, option, native_dir=False)
        except ValueError:
            continue
        kept[key] = value
    return kept


def _ATTENTION_ID(name: str) -> str:
    from ..options import _ATTENTION_IDS

    return _ATTENTION_IDS.get(name, name)


def _carry_over(prompt: dict, loader_id: str, settings: dict) -> list[str]:
    """Put the dropped nodes' speed-ups on the loader's SGLDOptions."""
    loader = prompt[loader_id]
    settings = _supported(_model_type(loader), settings)
    if not settings:
        return []
    link = loader["inputs"].get("sgld_options")
    if isinstance(link, list) and len(link) == 2:
        options = prompt.get(str(link[0]))
        if not options or options.get("class_type") != "SGLDOptions":
            return []  # options built some other way; leave them alone
    else:
        options_id = f"{loader_id}.sgld_options"
        options = prompt[options_id] = {"class_type": "SGLDOptions", "inputs": {}}
        loader["inputs"]["sgld_options"] = [options_id, 0]
    applied = []
    config = settings.pop("attention_backend_config", None)
    for key, value in settings.items():
        if options["inputs"].get(key, _UNSET[key]) == _UNSET[key]:
            options["inputs"][key] = value
            applied.append(f"{key}={value}")
    # Sol's parameters apply only when Sol is the backend that ends up selected.
    inputs = options["inputs"]
    if (
        config
        and inputs.get("attention_backend") == _SOL[1]
        and inputs.get("attention_backend_config", "") == ""
    ):
        inputs["attention_backend_config"] = config
        applied.append(f"attention_backend_config={config}")
    return applied


def _sol_config(node: dict) -> str:
    """The community Sol node's sparsity as SGLD sol_attn config.

    Those nodes run every step sparse with their own tau; SGLD's defaults keep
    the first 10 steps and layers 0-1 dense, which leaves a turbo (8-12 step)
    sampler almost dense.
    """
    inputs = node.get("inputs", {})
    config = {"dense_steps": 0, "dense_layers": ""}
    tau = inputs.get("tau", inputs.get("tau_start"))
    if isinstance(tau, (int, float)):
        config["tau"] = float(tau)
    if isinstance(inputs.get("thresh_type"), str):
        config["thresh_type"] = inputs["thresh_type"]
    dense = inputs.get("dense_blocks")
    # SGLD takes non-negative indices and ranges only ("0-3,47").
    if (
        isinstance(dense, str)
        and re.fullmatch(r"[\d,\s-]*", dense)
        and not re.search(r"(^|,)\s*-", dense)
    ):
        config["dense_layers"] = dense
    return json.dumps(config, sort_keys=True)


def drop_speed_patches(prompt: dict) -> list[str]:
    """Rewrite patch / LoRA nodes on SGLD models in place; return what changed."""
    dropped, carry = [], {}
    for node_id in list(prompt):
        node = prompt.get(node_id)
        cls = node.get("class_type") if node else None
        if cls not in SPEED_ONLY_PATCHES and cls not in CUSTOM_LORA_NODES:
            continue
        source = node.get("inputs", {}).get("model")
        if not (isinstance(source, list) and len(source) == 2):
            continue
        loader_id = _sgld_loader(prompt, str(source[0]), set())
        if loader_id is None:
            continue
        if cls in CUSTOM_LORA_NODES:
            name_key, strength_key = CUSTOM_LORA_NODES[cls]
            inputs = node["inputs"]
            node["class_type"] = "SGLDLoraLoader"
            node["inputs"] = {
                "model": source,
                "lora_name": inputs[name_key],
                "strength_model": inputs.get(strength_key, 1.0),
                "nickname": "",
                "target": "all",
            }
            dropped.append(f"{node_id}:{cls}->SGLDLoraLoader")
            continue
        consumers = [
            (other, key)
            for other in prompt.values()
            for key, value in other.get("inputs", {}).items()
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == node_id
        ]
        if any(other["inputs"][key][1] != 0 for other, key in consumers):
            continue  # an extra output is used; leave the node alone
        for other, key in consumers:
            other["inputs"][key] = list(source)
        del prompt[node_id]
        dropped.append(f"{node_id}:{node['class_type']}")
        setting = _sgld_setting(node)
        if setting:
            # The first speed-up per kind wins, as on ComfyUI's patch chain.
            settings = carry.setdefault(loader_id, {})
            if setting == _SOL and "attention_backend" not in settings:
                settings["attention_backend_config"] = _sol_config(node)
            settings.setdefault(*setting)
    for loader_id, settings in carry.items():
        applied = _carry_over(prompt, loader_id, settings)
        if applied:
            dropped.append(f"{loader_id}:SGLDOptions({', '.join(applied)})")
    return dropped


def install() -> None:
    try:
        from server import PromptServer
    except ImportError:  # not running inside ComfyUI
        return
    if PromptServer.instance is None:
        return

    def on_prompt(json_data):
        prompt = json_data.get("prompt")
        if isinstance(prompt, dict):
            dropped = drop_speed_patches(prompt)
            if dropped:
                logger.warning(
                    "SGLD: rewrote nodes on the SGLD model (speed-only patches replaced "
                    "by the matching SGLDOptions setting; custom LoRA loaders routed to "
                    "the worker): %s",
                    ", ".join(dropped),
                )
        return json_data

    PromptServer.instance.add_on_prompt_handler(on_prompt)
