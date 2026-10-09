"""Single DiT steps through the SGLD executor at the template's stage shapes.

env: LTX_UNET (diffusion_models file), LTX_CONDS (ref_encode output), SGLD_OPTS (json),
     SGLD_LORA / SGLD_LORA_STRENGTH, MGPU_OUT (tensors/<name>.pt), MGPU_REPS
Saves outputs for compare_mgpu.py and prints per-case step times.
"""

import json
import os
import statistics
import sys
import time

import torch

sys.argv = [sys.argv[0]]
COMFY = "/scratch/data/sgld_comfy/ComfyUI-ltx25"
os.chdir(COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
import folder_paths  # noqa: E402

from ComfyUI_SGLDiffusion.core.generator import SGLDiffusionGenerator  # noqa: E402

RES = os.environ["LTX_RES"]
UNET = os.environ["LTX_UNET"]
CONDS = os.path.join(RES, "tensors", os.environ["LTX_CONDS"])
OPTS = json.loads(os.environ.get("SGLD_OPTS", "{}"))
OUT = os.path.join(RES, "tensors", os.environ["MGPU_OUT"])
REPS = int(os.environ.get("MGPU_REPS", 3))
FPS = 25.0
# Template stage shapes for 1280x704, 121 frames: latent 16 frames, 32x spatial.
CASES = {
    "stage1": dict(T=16, H=11, W=20, sigma=0.909375),
    "stage2": dict(T=16, H=22, W=40, sigma=0.725),
    "stage1_i2v": dict(T=16, H=11, W=20, sigma=0.909375, i2v=True),
    # Same math as stage1 at batch 2: the kernel-shape noise floor of one config.
    "stage1_b2": dict(T=16, H=11, W=20, sigma=0.909375, batch=2),
    "odd_tokens": dict(T=15, H=11, W=21, sigma=0.909375, reps=1),
}
TA = 121  # audio latent frames for 121 video frames at 25 fps


def main():
    conds = torch.load(CONDS)
    path = folder_paths.get_full_path("diffusion_models", UNET)
    gen = SGLDiffusionGenerator.shared()
    t0 = time.time()
    gen.load_model(path, model_options={}, sgld_options=OPTS)
    log = {"opts": OPTS, "unet": UNET, "load_s": round(time.time() - t0, 1)}
    print("load %.1fs" % log["load_s"], flush=True)
    ex = gen.executor
    lora = os.environ.get("SGLD_LORA")
    if lora:
        t0 = time.time()
        ex.set_lora(
            lora_nickname=["distilled"],
            lora_path=[folder_paths.get_full_path("loras", lora)],
            strength=[float(os.environ.get("SGLD_LORA_STRENGTH", 1.0))],
            target=["all"],
        )
        log["set_lora_s"] = round(time.time() - t0, 1)
        print("set_lora %.1fs" % log["set_lora_s"], flush=True)
    dev = torch.device("cuda")
    bf = torch.bfloat16
    saved = {}
    with torch.no_grad():
        context = ex.preprocess_text_embeds(conds["pos"].to(dev, bf), unprocessed=True)
        saved["context"] = context.float().cpu()
        for name, case in CASES.items():
            g = torch.Generator().manual_seed(0)
            T, H, W = case["T"], case["H"], case["W"]
            video = torch.randn(1, 128, T, H, W, generator=g).to(dev, bf)
            audio = torch.randn(1, 8, TA, 16, generator=g).to(dev, bf)
            nb = case.get("batch", 1)
            ctx = context.expand(nb, -1, -1)
            video, audio = video.expand(nb, -1, -1, -1, -1), audio.expand(nb, -1, -1, -1)
            a_ts = torch.full((nb,), case["sigma"], device=dev)
            if case.get("i2v"):
                v_ts = torch.full((1, T * H * W, 1), case["sigma"], device=dev)
                v_ts[:, : H * W] = 0.0
                mask = torch.ones(1, 1, T, H, W, device=dev)
                mask[:, :, 0] = 0.0
                extra = {"denoise_mask": mask}
            else:
                v_ts = torch.full((nb,), case["sigma"], device=dev)
                extra = {}
            times = []
            try:
                for _ in range(case.get("reps", REPS)):
                    ex.begin_sampler_run()
                    torch.cuda.synchronize()
                    t0 = time.time()
                    out = ex.forward([video, audio], (v_ts, a_ts), ctx,
                                     attention_mask=None, frame_rate=FPS,
                                     transformer_options={}, **extra)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
            except Exception as exc:  # noqa: BLE001 - recorded as evidence
                msg = str(exc).splitlines()[0][:400]
                print(f"{name}: ERROR {type(exc).__name__}: {msg}", flush=True)
                log[name] = {"error": f"{type(exc).__name__}: {msg}"}
                continue
            saved[name + "_video"] = out[0][:1].float().cpu()
            saved[name + "_audio"] = out[1][:1].float().cpu()
            log[name] = {"times": [round(t, 3) for t in times],
                         "warm_median": round(statistics.median(times[1:] or times), 3),
                         "tokens": T * H * W}
            print(f"{name}: tokens={T*H*W} times={log[name]['times']}", flush=True)
    torch.save(saved, OUT)
    with open(OUT.replace(".pt", ".json"), "w") as f:
        json.dump(log, f, indent=1)
    gen.close_generator()


if __name__ == "__main__":
    main()
