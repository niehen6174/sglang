"""Drop speed-only model patch nodes that sit on an SGLD model.

The DiT runs in the SGLD worker, which picks its own attention kernel and step
caching (SGLDOptions), so attention / sparse-attention / cache / compile patch
nodes cannot change anything there. Many of them also refuse a model that is not
ComfyUI's native class. Before the prompt is queued, each such node whose MODEL
input comes from an SGLD loader is removed and its consumers are wired to its
input. Nodes that change what the model computes (ControlNet, block patches)
are left in place, so they still fail loudly at sampling time.
"""

import logging

logger = logging.getLogger(__name__)

SGLD_LOADERS = {"SGLDUNETLoader", "SGLDNativeModelLoader"}

# MODEL -> MODEL nodes that only change speed / memory of the DiT forward.
SPEED_ONLY_PATCHES = {
    # ComfyUI core
    "ModelAttentionBackend",
    "BlockSparseAttention",
    "EasyCache",
    "LazyCache",
    "TorchCompileModel",
    # KJNodes
    "PathchSageAttentionKJ",
    "MiniMaxH3MemoryEfficientSageAttentionPatch",
    "TorchCompileModelQwenImage",
    "TorchCompileModelAdvanced",
    "ModelPatchTorchSettings",
    "MiniMaxLowVRAMAttention",
    "MiniMaxChunkFeedForward",
    # MiniMax H3 community
    "SolAttnMiniMax",
    "MiniMaxH3ScheduledSolAttentionPatch",
    "VedaSparseAttention",
    "SpectrumApplyMiniMaxH3",
    "H3Cache",
    "H3BlockCache",
    # Qwen-Image 2.1 T8 pack
    "QwenImage21SageAttentionT8",
    "QwenImage21BlockCacheT8",
    "QwenImage21SpectrumT8",
    # generic caches
    "TeaCache",
    "MagCache",
    "ApplyFBCacheOnModel",
}


# MODEL-only LoRA nodes that patch DiT internals themselves (so they bypass
# comfy.sd.load_lora_for_models): class -> (lora file input, strength input).
CUSTOM_LORA_NODES = {
    "MiniMaxH3TurboLoRA": ("lora_name", "strength"),
    "LoraLoaderBypassModelOnly": ("lora_name", "strength_model"),
}


def _from_sgld_loader(prompt: dict, node_id: str, seen: set) -> bool:
    """Whether the MODEL reaching ``node_id`` was loaded by an SGLD loader."""
    while node_id in prompt and node_id not in seen:
        seen.add(node_id)
        node = prompt[node_id]
        if node.get("class_type") in SGLD_LOADERS:
            return True
        model = node.get("inputs", {}).get("model")
        if not (isinstance(model, list) and len(model) == 2):
            return False
        node_id = str(model[0])
    return False


def drop_speed_patches(prompt: dict) -> list[str]:
    """Rewrite patch / LoRA nodes on SGLD models in place; return what changed."""
    dropped = []
    for node_id in list(prompt):
        node = prompt.get(node_id)
        cls = node.get("class_type") if node else None
        if cls not in SPEED_ONLY_PATCHES and cls not in CUSTOM_LORA_NODES:
            continue
        source = node.get("inputs", {}).get("model")
        if not (isinstance(source, list) and len(source) == 2):
            continue
        if not _from_sgld_loader(prompt, str(source[0]), set()):
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
                    "SGLD: rewrote nodes on the SGLD model (speed-only patches dropped, "
                    "SGLD sets attention and caching itself via SGLDOptions; custom "
                    "LoRA loaders routed to the worker): %s",
                    ", ".join(dropped),
                )
        return json_data

    PromptServer.instance.add_on_prompt_handler(on_prompt)
