#!/usr/bin/env python3
"""Per-step DiT latency: ComfyUI native (prefix cache on) vs SGLD executor.

Uses the t2i_b1_64x64 / edit1_b1_64x64 records captured by parity_dit.py.
Each mode runs `steps` calls with the same inputs after 3 warmup calls in a
fresh sampler run (step 1 prefills the prefix cache; later steps reuse it).
"""

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "/scratch/data/sgld_comfy/ComfyUI-qwen21")
DIT = "/scratch/data/sgld_comfy/ComfyUI-qwen21/models/diffusion_models/qwen_image_2.1_bf16.safetensors"


def _kw(st, sig, cache_device="auto"):
    kw = {"transformer_options": {"sample_sigmas": sig, "qwen_image21_cache": {"device": cache_device}}}
    if st["ref_latents"]:
        kw["ref_latents"] = [r.cuda() for r in st["ref_latents"]]
    if st["image_slots"] is not None:
        kw["image_slots"] = st["image_slots"]
    return kw


def bench(fn, steps):
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["comfy", "sgld"])
    p.add_argument("--capture", default="/scratch/data/sgld_comfy/tmp/qi21_parity_capture.pt")
    p.add_argument("--cases", nargs="*", default=["t2i_b1_64x64", "edit1_b1_64x64"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--sgld-options", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--dit", default=DIT)
    p.add_argument("--cache-device", default="auto")
    args = p.parse_args()
    data = torch.load(args.capture)
    recs = {r["name"]: r for r in data["records"]}
    sig = torch.tensor(data["sigmas"] + [0.0])
    res = {}
    if args.mode == "comfy":
        import comfy.model_management as mm
        import comfy.sd

        model = comfy.sd.load_diffusion_model(args.dit)
        mm.load_models_gpu([model])
        dm = model.model.diffusion_model
        for name in args.cases:
            st = recs[name]["steps"][1]
            x, t, c = st["x"].cuda(), st["timestep"].cuda(), st["context"].cuda()
            model.model.current_patcher = model  # enables the prefix K/V cache like pre_run
            kw = _kw(st, sig)
            with torch.inference_mode():
                prefill = bench(lambda: dm(x, t, c, **kw), 1)
                times = bench(lambda: dm(x, t, c, **kw), args.steps)
            model.model.current_patcher = None
            res[name] = {"prefill_s": prefill[0], "step_ms_median": sorted(times)[len(times) // 2] * 1e3}
            print(name, res[name], flush=True)
    else:
        from comfy import model_management  # noqa: F401

        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import (
            SGLDiffusionGenerator,
        )

        gen = SGLDiffusionGenerator.shared()
        gen.load_model(args.dit, sgld_options=json.loads(args.sgld_options) if args.sgld_options else None)
        ex = gen.executor
        try:
            for name in args.cases:
                st = recs[name]["steps"][1]
                x, t, c = st["x"].cuda(), st["timestep"].cuda(), st["context"].cuda()
                kw = _kw(st, sig, args.cache_device)
                for _ in range(2):  # second run: compile / autotune caches warm
                    ex.begin_sampler_run()
                    with torch.no_grad():
                        prefill = bench(lambda: ex(x, t, c, **kw), 1)
                        times = bench(lambda: ex(x, t, c, **kw), args.steps)
                    ex.end_sampler_run()
                res[name] = {"prefill_s": prefill[0], "step_ms_median": sorted(times)[len(times) // 2] * 1e3}
                print(name, res[name], flush=True)
        finally:
            gen.close_generator()
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
