"""Encode a positive / negative prompt with ComfyUI's LTX-2.5 Gemma4 TE and save raw conds."""

import os
import sys

import torch

sys.argv = [sys.argv[0]]
os.chdir("/scratch/data/sgld_comfy/ComfyUI-ltx25")
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402

OUT = os.path.join(os.environ["LTX_RES"], "tensors", os.environ.get("LTX_CONDS", "conds.pt"))
POS = (
    "A close-up of an Arctic hunter's face, frost dusting his dark beard, the camera "
    "slowly pulls back revealing a polar bear on a distant ice ridge. The wind carries "
    "the distant sound of shifting ice, a single low growl rolling across the water."
)
NEG = "pc game, console game, video game, cartoon, childish, ugly"

# LTX_TE="folder:name,folder:name" (LTX-2.3: text_encoders gemma3 + checkpoints all-in-one)
spec = os.environ.get(
    "LTX_TE", "text_encoders:gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
)
paths = [folder_paths.get_full_path(*item.split(":", 1)) for item in spec.split(",")]
clip = comfy.sd.load_clip(ckpt_paths=paths, clip_type=comfy.sd.CLIPType.LTXV)
out = {}
for name, text in (("pos", POS), ("neg", NEG)):
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    c, meta = cond[0]
    print(name, tuple(c.shape), c.dtype, {k: v for k, v in meta.items() if not torch.is_tensor(v)})
    out[name] = c.cpu()
    out[name + "_meta"] = {k: v for k, v in meta.items() if not torch.is_tensor(v)}
torch.save(out, OUT)
print("saved", OUT)
