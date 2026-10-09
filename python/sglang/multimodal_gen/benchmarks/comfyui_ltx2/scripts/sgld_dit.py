"""Run the SGLD integrated executor on ref_dit.py inputs and compare with ComfyUI."""

import json
import os
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
UNET = os.environ.get(
    "LTX_UNET", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
)
REF = os.path.join(RES, "tensors", os.environ.get("LTX_REF_OUT", "ref_dit.pt"))
OPTS = json.loads(os.environ.get("SGLD_OPTS", "{}"))


def stats(name, ref, out):
    ref = ref.float().flatten()
    out = out.float().flatten().to(ref.device)
    diff = (out - ref).abs()
    cos = torch.nn.functional.cosine_similarity(ref, out, dim=0).item()
    rel = (diff.norm() / ref.norm().clamp(min=1e-12)).item()
    line = (
        f"{name:14s} max_abs={diff.max().item():.4e} mean_abs={diff.mean().item():.4e} "
        f"rel_l2={rel:.4e} cos={cos:.6f} ref_std={ref.std().item():.4f}"
    )
    print(line, flush=True)
    return {"max_abs": diff.max().item(), "mean_abs": diff.mean().item(), "rel_l2": rel, "cos": cos}


def main():
    ref = torch.load(REF)
    path = folder_paths.get_full_path("diffusion_models", UNET)
    gen = SGLDiffusionGenerator.shared()
    t0 = time.time()
    gen.load_model(path, model_options={}, sgld_options=OPTS)
    print("load %.1fs" % (time.time() - t0), flush=True)
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
        print("set_lora %.1fs" % (time.time() - t0), flush=True)
    dev = torch.device("cuda")
    bf = torch.bfloat16
    results = {}
    with torch.no_grad():
        for name in ("pos", "neg"):
            ctx = ex.preprocess_text_embeds(ref[f"raw_{name}"].to(dev, bf), unprocessed=True)
            results[f"ctx_{name}"] = stats(f"ctx_{name}", ref[f"ctx_{name}"], ctx)
        # Feed ComfyUI's own processed context so the DiT comparison is isolated.
        context = torch.cat([ref["ctx_pos"], ref["ctx_neg"]], 0).to(dev, bf)
        video = ref["video"].to(dev, bf)
        audio = ref["audio"].to(dev, bf)
        ts = torch.full((2,), ref["sigma"], device=dev)
        ex.begin_sampler_run()
        for i in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            out = ex.forward([video, audio], (ts, ts), context, attention_mask=None,
                             frame_rate=ref["fps"], transformer_options={})
            torch.cuda.synchronize()
            print("t2v step %d %.3fs" % (i, time.time() - t0), flush=True)
        saved = {"v_out": out[0].float().cpu(), "a_out": out[1].float().cpu()}
        results["t2v_video"] = stats("t2v_video", ref["v_out"], out[0])
        results["t2v_audio"] = stats("t2v_audio", ref["a_out"], out[1])

        ex.begin_sampler_run()
        a_ts = torch.full((1,), ref["sigma"], device=dev)
        out = ex.forward([video[:1], audio[:1]], (ref["i2v_v_ts"].to(dev), a_ts), context[:1],
                         attention_mask=None, frame_rate=ref["fps"], transformer_options={},
                         denoise_mask=ref["i2v_mask"].to(dev))
        saved.update(i2v_v_out=out[0].float().cpu(), i2v_a_out=out[1].float().cpu())
        if os.environ.get("SGLD_SAVE"):
            torch.save(saved, os.path.join(RES, "tensors", os.environ["SGLD_SAVE"]))
        results["i2v_video"] = stats("i2v_video", ref["i2v_v_out"], out[0])
        results["i2v_audio"] = stats("i2v_audio", ref["i2v_a_out"], out[1])

    with open(os.path.join(RES, "logs", os.environ.get("PARITY_JSON", "dit_parity.json")), "w") as f:
        json.dump({"opts": OPTS, "unet": UNET, **results}, f, indent=1)
    gen.close_generator()


if __name__ == "__main__":
    main()
