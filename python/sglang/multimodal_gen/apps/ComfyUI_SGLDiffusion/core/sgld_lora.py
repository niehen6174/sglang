"""Worker-side LoRA entries on an SGLD model.

The SGLD worker owns the DiT weights, so a LoRA on an SGLD model is an adapter
entry the worker loads, not a ComfyUI weight patch. SGLDLoraLoader and
SGLDLoraLoaderWithCLIP add them; other LoRA nodes are rewritten to these by the
workflow converter (``python -m sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.convert``).
"""

import hashlib
import logging

CONVERT_HINT = (
    "convert the workflow first: python -m "
    "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.convert <workflow.json>"
)


def add_sgld_lora(
    model, lora_path: str, strength: float, nickname: str = "", target: str = "all"
):
    """Clone ``model`` with one more worker-side LoRA adapter."""
    bi = model.clone()
    if not nickname:
        digest = hashlib.sha1(f"{lora_path}:{len(bi.patches)}".encode()).hexdigest()[
            :12
        ]
        nickname = f"lora_{digest}"
    bi.patches[nickname] = (lora_path, strength, target)
    lora_input = {"lora_nickname": [], "lora_path": [], "strength": [], "target": []}
    for name, (path, scale, tgt) in bi.patches.items():
        lora_input["lora_nickname"].append(name)
        lora_input["lora_path"].append(path)
        lora_input["strength"].append(scale)
        lora_input["target"].append(tgt)
    targets = set(lora_input["target"])
    if "all" in targets and len(targets) > 1:
        # set_lora groups adapters per target string; "all" overlapping a
        # named target makes the worker keep only one group on that module.
        raise ValueError(
            "Chained LoRA loaders mix target 'all' with "
            f"{sorted(targets - {'all'})}; use the same target on every node"
        )
    # The shared worker is bound to the selected MODEL at sampling time.
    bi.model_options["sgld_lora_input"] = lora_input
    return bi


class _DropUnetNotLoaded(logging.Filter):
    """The DiT half of a LoRA is not applied in ComfyUI; hide its 'not loaded' lines."""

    def filter(self, record):
        return "lora key not loaded" not in record.getMessage()


def apply_clip_lora(clip, lora_path: str, strength: float):
    """The text-encoder half of a LoRA, applied by ComfyUI as usual."""
    import comfy.sd
    import comfy.utils

    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    root = logging.getLogger()
    hide = _DropUnetNotLoaded()
    root.addFilter(hide)
    try:
        _, new_clip = comfy.sd.load_lora_for_models(None, clip, lora, 0, strength)
    finally:
        root.removeFilter(hide)
    return new_clip
