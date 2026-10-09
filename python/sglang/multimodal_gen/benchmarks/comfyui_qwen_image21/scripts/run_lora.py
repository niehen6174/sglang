#!/usr/bin/env python3
"""Qwen-Image-2.1 + LoRA through the ComfyUI /prompt API (official vs SGLD).

--loader official : UNETLoader -> LoraLoaderModelOnly            (ComfyUI native LoRA)
--loader sgld     : SGLDUNETLoader -> SGLDLoraLoader             (integrated)
--loader native_on_sgld : SGLDUNETLoader -> LoraLoaderModelOnly  (must fail loudly)
--strength 0 drops the LoRA node entirely (base model).
--sampler turbo6 : the viggle-turbo 6-step schedule (ManualSigmas, euler,
                   BasicGuider = no CFG), sigmas = ViggleTurboSigmas at 1024x1024.
--sampler base25 : KSampler euler/simple 25 steps cfg 1 (official template).
Peak VRAM is the max of nvidia-smi memory.used on the GPU (all processes)
sampled every 0.1 s while the prompt runs.
"""

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid

PROMPT = (
    "A studio portrait of an old fisherman mending a net, warm rim light, 85mm, "
    "detailed weathered skin, a small wooden sign on the wall reads \"OPEN\"."
)
TURBO_SIGMAS = "1.000000, 0.967754, 0.933358, 0.857192, 0.666756, 0.400096, 0.0"
LORA = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"


def build(args):
    g = {}
    unet = args.unet
    if args.loader == "official":
        g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
    else:
        g["1"] = {"class_type": "SGLDUNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
    model = ["1", 0]
    if args.strength > 0:
        if args.loader == "sgld":
            g["10"] = {
                "class_type": "SGLDLoraLoader",
                "inputs": {"model": model, "lora_name": args.lora, "strength_model": args.strength,
                           "nickname": args.nickname, "target": "all"},
            }
        else:
            g["10"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": model, "lora_name": args.lora, "strength_model": args.strength},
            }
        model = ["10", 0]
    g["2"] = {"class_type": "QwenImage21Cache", "inputs": {"model": model, "device": "auto", "dtype": "default"}}
    g["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                                                    "type": "qwen_image", "device": "default"}}
    g["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}}
    g["5"] = {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["3", 0], "prompt": args.prompt, "negative_prompt": "", "resolution": 1024}}
    g["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    if args.sampler == "turbo6":
        g["11"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": TURBO_SIGMAS}}
        g["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        g["13"] = {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["5", 0]}}
        g["14"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}}
        g["7"] = {"class_type": "SamplerCustomAdvanced",
                  "inputs": {"noise": ["14", 0], "guider": ["13", 0], "sampler": ["12", 0],
                             "sigmas": ["11", 0], "latent_image": ["6", 0]}}
    else:
        g["7"] = {"class_type": "KSampler",
                  "inputs": {"model": ["2", 0], "seed": args.seed, "steps": 25, "cfg": 1.0,
                             "sampler_name": "euler", "scheduler": "simple", "positive": ["5", 0],
                             "negative": ["5", 1], "latent_image": ["6", 0], "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "qi21_lora"}}
    return g


class PeakVram:
    def __init__(self, gpu):
        self.gpu, self.peak, self._stop = gpu, 0, False
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                  "-i", str(self.gpu)], capture_output=True, text=True).stdout
            try:
                self.peak = max(self.peak, int(out.strip()))
            except ValueError:
                pass
            time.sleep(0.1)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._t.join()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loader", choices=["official", "sgld", "native_on_sgld"], required=True)
    p.add_argument("--sampler", choices=["turbo6", "base25"], default="turbo6")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--lora", default=LORA)
    p.add_argument("--nickname", default="viggle_turbo")
    p.add_argument("--unet", default="qwen_image_2.1_bf16.safetensors")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", required=True)
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--expect-error", action="store_true")
    p.add_argument("--comfy-dir", default="/scratch/data/sgld_comfy/ComfyUI-qwen21")
    p.add_argument("--out-dir", default="/scratch/data/sgld_comfy/results/qwen_image21/images/lora")
    args = p.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    graph = build(args)
    with PeakVram(args.gpu) as vram:
        t0 = time.time()
        req = urllib.request.Request(base + "/prompt", data=json.dumps({"prompt": graph, "client_id": uuid.uuid4().hex}).encode(),
                                     headers={"Content-Type": "application/json"})
        pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
        while True:
            h = json.loads(urllib.request.urlopen(f"{base}/history/{pid}").read())
            if pid in h and (h[pid]["status"].get("completed") or h[pid]["status"].get("status_str") == "error"):
                break
            time.sleep(0.2)
        dt = time.time() - t0
    st = h[pid]["status"]
    rec = {"tag": args.tag, "loader": args.loader, "sampler": args.sampler, "strength": args.strength,
           "seed": args.seed, "unet": args.unet, "status": st.get("status_str"), "seconds": round(dt, 2),
           "peak_vram_mib": vram.peak}
    if st.get("status_str") != "success":
        err = [m for m in st.get("messages", []) if m[0] == "execution_error"]
        if err:
            rec["error"] = {"node_type": err[0][1].get("node_type"), "exception_type": err[0][1].get("exception_type"),
                            "message": err[0][1].get("exception_message", "")[:600]}
        print(json.dumps(rec))
        if not args.expect_error:
            raise SystemExit(1)
        return
    if args.expect_error:
        print(json.dumps(rec))
        raise SystemExit("expected an error but the prompt succeeded")
    img = h[pid]["outputs"]["9"]["images"][0]
    src = os.path.join(args.comfy_dir, "output", img.get("subfolder", ""), img["filename"])
    os.makedirs(args.out_dir, exist_ok=True)
    rec["image"] = os.path.join(args.out_dir, f"{args.tag}.png")
    shutil.copy(src, rec["image"])
    os.remove(src)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
