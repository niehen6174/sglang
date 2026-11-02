"""Route ComfyUI LoRA loading for SGLD models to the SGLD worker.

Every common LoRA node (core LoraLoader / LoraLoaderModelOnly, rgthree Power Lora,
AusBoss, LoraManager, Pixaroma, Easy-Use stacks) ends in
``comfy.sd.load_lora_for_models``. For an SGLD model the worker owns the DiT
weights, so the model half of that call is turned into an SGLD adapter entry and
only the CLIP half runs in ComfyUI.
"""

import hashlib
import logging
import os
import tempfile
from collections import OrderedDict

logger = logging.getLogger(__name__)

# id(state dict returned by load_torch_file) -> (path, key count). Ids can be
# reused once a dict is freed, so the key count is checked on lookup.
_LOADED_PATHS: "OrderedDict[int, tuple[str, int]]" = OrderedDict()
_MAX_TRACKED = 64
_installed = False


def add_sgld_lora(model, lora_path: str, strength: float, nickname: str = "", target: str = "all"):
    """Clone ``model`` with one more worker-side LoRA adapter."""
    bi = model.clone()
    if not nickname:
        digest = hashlib.sha1(f"{lora_path}:{len(bi.patches)}".encode()).hexdigest()[:12]
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


def _lora_file(lora: dict) -> str:
    """Path of the file ``lora`` was loaded from, or a cached copy of it."""
    hit = _LOADED_PATHS.get(id(lora))
    if hit is not None and hit[1] == len(lora):
        return hit[0]
    from safetensors.torch import save_file

    digest = hashlib.sha1()
    for key in sorted(lora):
        tensor = lora[key]
        digest.update(f"{key}{tuple(tensor.shape)}{tensor.dtype}".encode())
        digest.update(tensor.reshape(-1)[:64].float().cpu().numpy().tobytes())
    path = os.path.join(tempfile.gettempdir(), "sgld_lora", f"{digest.hexdigest()}.safetensors")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file({k: v.contiguous() for k, v in lora.items()}, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


class _DropUnetNotLoaded(logging.Filter):
    """The CLIP-only pass reports every DiT key as NOT LOADED; those go to the worker."""

    def filter(self, record):
        return not str(record.getMessage()).startswith("NOT LOADED")


def install() -> None:
    global _installed
    if _installed:
        return
    try:
        import comfy.sd
        import comfy.utils

        from .model_patcher import SGLDModelPatcher
    except ImportError:  # not running inside ComfyUI
        return
    _installed = True
    orig_load_torch_file = comfy.utils.load_torch_file
    orig_load_lora_for_models = comfy.sd.load_lora_for_models

    def load_torch_file(ckpt, *args, **kwargs):
        out = orig_load_torch_file(ckpt, *args, **kwargs)
        sd = out[0] if isinstance(out, tuple) else out
        if isinstance(sd, dict) and isinstance(ckpt, str):
            _LOADED_PATHS[id(sd)] = (ckpt, len(sd))
            while len(_LOADED_PATHS) > _MAX_TRACKED:
                _LOADED_PATHS.popitem(last=False)
        return out

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        if not isinstance(model, SGLDModelPatcher):
            return orig_load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata)
        new_model = model
        if strength_model != 0:
            new_model = add_sgld_lora(model, _lora_file(lora), strength_model)
        new_clip = None
        if clip is not None:
            root = logging.getLogger()
            flt = _DropUnetNotLoaded()
            root.addFilter(flt)
            try:
                _, new_clip = orig_load_lora_for_models(None, clip, lora, 0, strength_clip, lora_metadata)
            finally:
                root.removeFilter(flt)
        return new_model, new_clip

    comfy.utils.load_torch_file = load_torch_file
    comfy.sd.load_lora_for_models = load_lora_for_models
