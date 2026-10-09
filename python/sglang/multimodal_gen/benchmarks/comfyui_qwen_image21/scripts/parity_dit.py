#!/usr/bin/env python3
"""DiT parity: ComfyUI native QwenImage21 vs the SGLD integrated executor.

Phase ``comfy``: encode real prompts (and reference images) with ComfyUI's
TextEncodeQwenImage21, run ComfyUI's native DiT through ``apply_model`` and
capture the exact ``diffusion_model`` kwargs and raw outputs.

Phase ``sgld``: start the SGLD worker through the plugin's
SGLDiffusionGenerator, replay the captured kwargs through the executor (one
sampler run per case, so later sigmas hit the worker's prefix K/V cache) and
compare.
"""

import argparse
import json
import math
import time
import os
import sys

import numpy as np
import torch
from PIL import Image

COMFY = "/scratch/data/sgld_comfy/ComfyUI-qwen21"
sys.path.insert(0, COMFY)
MODELS = f"{COMFY}/models"
DIT = f"{MODELS}/diffusion_models/qwen_image_2.1_bf16.safetensors"
TE = f"{MODELS}/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
VAE = f"{MODELS}/vae/qwen_image_2.1_vae_bf16.safetensors"

POS = (
    "A red fox sitting in a snowy birch forest at golden hour, detailed fur, "
    "soft rim light, shallow depth of field, photorealistic."
)
NEG = "blurry, low quality, distorted anatomy, watermark, text, oversaturated colors"
EDIT = "Make the fox wear a small green knitted scarf and turn the snow into autumn leaves."
SIGMAS = [1.0, 0.75, 0.4, 0.1]


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None]


def cases(ref_paths):
    # (name, batch rows, target latent hw, ref images)
    yield "t2i_b1_64x64", ["pos"], (64, 64), []
    yield "t2i_b1_63x80", ["pos"], (63, 80), []
    yield "cfg_b2_64x64", ["neg", "pos"], (64, 64), []
    if ref_paths:
        yield "edit1_b1_64x64", ["pos"], (64, 64), ref_paths[:1]
        yield "edit1_b1_63x63", ["pos"], (63, 63), ref_paths[:1]
        yield "edit2_b2_64x48", ["neg", "pos"], (64, 48), ref_paths[:2]


def encode(clip, vae, refs):
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21

    images = {f"image_{i + 1}": load_image(p) for i, p in enumerate(refs)}
    out = TextEncodeQwenImage21.execute(
        clip,
        EDIT if refs else POS,
        NEG,
        vae=vae if refs else None,
        resolution=1024,
        images=images,
    )
    pos, neg = out.result[0], out.result[1]
    return {"pos": pos, "neg": neg}


def run_comfy(args):
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils

    torch.manual_seed(0)
    clip = comfy.sd.load_clip([TE], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
    model = comfy.sd.load_diffusion_model(args.dit)
    base = model.model
    captured = {}

    def pre_hook(module, a, kw):
        captured["args"] = a
        captured["kwargs"] = kw

    def post_hook(module, a, kw, out):
        captured["out"] = out.detach().float().cpu()

    dm = base.diffusion_model
    dm.register_forward_pre_hook(pre_hook, with_kwargs=True)
    dm.register_forward_hook(post_hook, with_kwargs=True)

    records = []
    conds_cache = {}
    for name, rows, (h, w), refs in cases(args.refs):
        key = tuple(refs)
        if key not in conds_cache:
            conds_cache[key] = encode(clip, vae, refs)
    clean = base.process_latent_in(vae.encode(load_image(args.x0_image)[..., :3])).float().cpu()
    mm.unload_all_models()
    mm.soft_empty_cache()
    for name, rows, (h, w), refs in cases(args.refs):
        conds = conds_cache[tuple(refs)]
        ctxs, extra = [], {}
        for r in rows:
            c, d = conds[r][0]
            ctxs.append(c)
            extra = d
        L = min(c.shape[1] for c in ctxs)
        if len(rows) > 1:
            # ComfyUI only batches equal-length conds; crop to exercise B=2.
            ctxs = [c[:, :L] for c in ctxs]
        context = torch.cat(ctxs, 0)
        slots = extra.get("image_slots")
        if slots is not None and len(rows) > 1:
            slots = [min(s, L) for s in slots]
        ref_latents = extra.get("reference_latents")
        mm.load_models_gpu([model])
        dev = mm.get_torch_device()
        g = torch.Generator().manual_seed(1234)
        # A real image latent (normalized) as x0 keeps late-sigma inputs on-distribution.
        x0 = torch.nn.functional.interpolate(clean, size=(h, w), mode="bilinear").expand(len(rows), -1, -1, -1)
        noise = torch.randn(len(rows), 64, h, w, generator=g)
        steps = []
        for sigma in SIGMAS:
            x = ((1 - sigma) * x0 + sigma * noise).to(dev)
            t = torch.full((len(rows),), sigma, device=dev)
            kw = {"c_crossattn": context.to(dev)}
            if ref_latents is not None:
                kw["ref_latents"] = [base.process_latent_in(r).to(dev).expand(len(rows), -1, -1, -1) for r in ref_latents]
            if slots is not None:
                kw["image_slots"] = slots
            # Noise floor: ComfyUI's own eager (training) path vs its fused kernels.
            mm.in_training = True
            try:
                base.apply_model(x, t, transformer_options={}, **kw)
            finally:
                mm.in_training = False
            out_eager = captured["out"]
            base.apply_model(x, t, transformer_options={}, **kw)
            a, k = captured["args"], captured["kwargs"]
            steps.append(
                {
                    "x": a[0].detach().cpu(),
                    "timestep": a[1].detach().cpu(),
                    "context": k["context"].detach().cpu(),
                    "ref_latents": [r.detach().cpu() for r in (k.get("ref_latents") or [])],
                    "image_slots": k.get("image_slots"),
                    "out": captured["out"],
                    "out_eager": out_eager,
                }
            )
        records.append({"name": name, "steps": steps})
        print(name, "context", tuple(context.shape), "slots", slots, "refs", [tuple(r.shape) for r in (ref_latents or [])], flush=True)
    torch.save({"records": records, "sigmas": SIGMAS}, args.capture)


def metrics(ref, out):
    ref, out = ref.float().flatten(), out.float().flatten()
    diff = (ref - out).abs()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_l2": float((ref - out).norm() / ref.norm()),
        "cosine": float(torch.nn.functional.cosine_similarity(ref, out, dim=0)),
    }


def run_sgld(args):
    from comfy import model_management as mm  # noqa: F401  (initializes comfy device)

    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
        SGLDiffusionGenerator,
    )

    data = torch.load(args.capture)
    gen = SGLDiffusionGenerator.shared()
    opts = json.loads(args.sgld_options) if args.sgld_options else None
    gen.load_model(args.dit, sgld_options=opts)
    ex = gen.executor
    results = {}
    saved = {}
    try:
        for rec in data["records"]:
            ex.begin_sampler_run()
            rows = []
            sig = torch.tensor(data["sigmas"] + [0.0])
            for i, st in enumerate(rec["steps"]):
                kw = {"transformer_options": {"sample_sigmas": sig, "qwen_image21_cache": {"device": args.cache_device}}}
                if st["ref_latents"]:
                    kw["ref_latents"] = [r.cuda() for r in st["ref_latents"]]
                if st["image_slots"] is not None:
                    kw["image_slots"] = st["image_slots"]
                try:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = ex(st["x"].cuda(), st["timestep"].cuda(), st["context"].cuda(), **kw)
                    torch.cuda.synchronize()
                    step_ms = (time.perf_counter() - t0) * 1e3
                except RuntimeError as exc:
                    print(rec["name"], "ERROR", str(exc)[:400], flush=True)
                    rows.append({"sigma": data["sigmas"][i], "error": str(exc)[:400]})
                    break
                saved.setdefault(rec["name"], []).append(out.cpu())
                m = metrics(st["out"], out.cpu())
                m["sigma"] = data["sigmas"][i]
                m["call_ms"] = step_ms
                floor = metrics(st["out"], st["out_eager"])
                m["floor_rel_l2"] = floor["rel_l2"]
                m["floor_cosine"] = floor["cosine"]
                m["floor_max_abs"] = floor["max_abs"]
                if out.shape[0] > 1:
                    m["rows_rel_l2"] = [
                        metrics(st["out"][r], out[r].cpu())["rel_l2"] for r in range(out.shape[0])
                    ]
                rows.append(m)
                print(rec["name"], json.dumps(m), flush=True)
            ex.end_sampler_run()
            results[rec["name"]] = rows
    finally:
        gen.close_generator()
    with open(args.report, "w") as f:
        json.dump(results, f, indent=1)
    if args.save_out:
        torch.save(saved, args.save_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["comfy", "sgld"])
    p.add_argument("--capture", default="/scratch/data/sgld_comfy/tmp/qi21_parity_capture.pt")
    p.add_argument("--report", default="/scratch/data/sgld_comfy/results/qwen_image21/parity_dit.json")
    p.add_argument("--refs", nargs="*", default=[])
    p.add_argument("--sgld-options", default=None)
    p.add_argument("--cache-device", default="auto", help="QwenImage21Cache device: auto/gpu/cpu/off")
    p.add_argument("--save-out", default=None)
    p.add_argument("--dit", default=DIT)
    p.add_argument("--x0-image", default="/scratch/data/sgld_comfy/results/qwen_image21/inputs/ref_portrait.png")
    args = p.parse_args()
    if args.phase == "comfy":
        with torch.inference_mode():
            run_comfy(args)
    else:
        run_sgld(args)


if __name__ == "__main__":
    main()
