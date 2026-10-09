"""Run ComfyUI's native LTXAV DiT on fixed inputs and save outputs for parity checks.

Cases:
  t2v      : batch 2 (pos / neg), scalar sigma for video and audio
  i2v      : batch 1, video timestep per token with latent frame 0 = 0 (LTXVImgToVideo
             style denoise mask), audio sigma scalar
Also saves processed contexts (connector output) and a few block outputs.
"""

import os
import sys
import time

import torch

sys.argv = [sys.argv[0]]
os.chdir("/scratch/data/sgld_comfy/ComfyUI-ltx25")
if os.environ.get("LTX_DYNAMIC"):
    # Same DynamicVRAM setup as ComfyUI main.py (the server default on NVIDIA).
    import comfy_aimdo.control

    comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True)
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402

if os.environ.get("LTX_DYNAMIC"):
    import comfy.memory_management
    import comfy.model_patcher

    assert comfy_aimdo.control.init_devices(
        (d.index, 0) for d in mm.get_all_torch_devices()
    )
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True

RES = os.environ["LTX_RES"]
UNET = os.environ.get(
    "LTX_UNET", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
)
OUT = os.path.join(RES, "tensors", os.environ.get("LTX_REF_OUT", "ref_dit.pt"))
T, H, W = int(os.environ.get("LTX_T", 4)), int(os.environ.get("LTX_H", 8)), int(
    os.environ.get("LTX_W", 12)
)
TA = int(os.environ.get("LTX_TA", 26))
SIGMA = float(os.environ.get("LTX_SIGMA", 0.909375))
FPS = 24.0

conds = torch.load(os.path.join(RES, "tensors", os.environ.get("LTX_CONDS", "conds.pt")))
path = folder_paths.get_full_path("diffusion_models", UNET)
patcher = comfy.sd.load_diffusion_model(path)
LORA = os.environ.get("LTX_LORA")
if LORA:
    # Same as LoraLoaderModelOnly: patch the MODEL only.
    import comfy.utils

    lora_sd = comfy.utils.load_torch_file(folder_paths.get_full_path("loras", LORA), safe_load=True)
    patcher, _ = comfy.sd.load_lora_for_models(
        patcher, None, lora_sd, float(os.environ.get("LTX_LORA_STRENGTH", 1.0)), 0
    )
    print("lora patches", len(patcher.patches))
mm.load_models_gpu([patcher])
model = patcher.model
dm = model.diffusion_model
dev = mm.get_torch_device()
dtype = model.get_dtype_inference()
print("dtype", dtype, type(dm).__name__)

with torch.no_grad():
    ctx = {}
    for name in ("pos", "neg"):
        raw = conds[name].to(dev, dtype)
        ctx[name] = dm.preprocess_text_embeds(raw, unprocessed=True)
        print(name, "ctx", tuple(ctx[name].shape))

    g = torch.Generator("cpu").manual_seed(1234)
    video = torch.randn(2, 128, T, H, W, generator=g).to(dev, dtype)
    audio = torch.randn(2, 8, TA, 16, generator=g).to(dev, dtype)
    context = torch.cat([ctx["pos"], ctx["neg"]], 0)
    ts = torch.full((2,), SIGMA, device=dev, dtype=torch.float32)

    captured = {}
    hooks = []
    for idx in (0, 23, len(dm.transformer_blocks) - 1):
        def hook(_m, _i, out, idx=idx):
            captured[f"block{idx}_v"] = out[0].detach().float().cpu()
            captured[f"block{idx}_a"] = out[1].detach().float().cpu()
        hooks.append(dm.transformer_blocks[idx].register_forward_hook(hook))

    torch.cuda.synchronize()
    t0 = time.time()
    out = dm(
        [video.clone(), audio.clone()],
        (ts, ts),
        context=context,
        attention_mask=None,
        frame_rate=FPS,
        transformer_options={},
    )
    torch.cuda.synchronize()
    print("t2v forward %.3fs" % (time.time() - t0))
    for h in hooks:
        h.remove()
    v_out, a_out = out
    print("out", tuple(v_out.shape), tuple(a_out.shape))

    # I2V-style: per-token video timestep, first latent frame clean.
    mask = torch.ones(1, 1, T, H, W, device=dev)
    mask[:, :, 0] = 0.0
    v_ts = dm.patchifier.patchify((mask * SIGMA)[:, :1])[0]  # [1, N, 1]
    a_ts = torch.full((1,), SIGMA, device=dev, dtype=torch.float32)
    out_i2v = dm(
        [video[:1].clone(), audio[:1].clone()],
        (v_ts.float(), a_ts),
        context=ctx["pos"],
        attention_mask=None,
        frame_rate=FPS,
        transformer_options={},
        denoise_mask=mask,
    )

save = {
    "video": video.float().cpu(),
    "audio": audio.float().cpu(),
    "sigma": SIGMA,
    "fps": FPS,
    "raw_pos": conds["pos"],
    "raw_neg": conds["neg"],
    "ctx_pos": ctx["pos"].float().cpu(),
    "ctx_neg": ctx["neg"].float().cpu(),
    "v_out": v_out.float().cpu(),
    "a_out": a_out.float().cpu(),
    "i2v_mask": mask.cpu(),
    "i2v_v_ts": v_ts.float().cpu(),
    "i2v_v_out": out_i2v[0].float().cpu(),
    "i2v_a_out": out_i2v[1].float().cpu(),
    **captured,
}
torch.save(save, OUT)
print("saved", OUT)
