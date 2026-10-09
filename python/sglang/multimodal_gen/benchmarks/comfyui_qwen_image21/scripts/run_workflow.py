#!/usr/bin/env python3
"""Drive ComfyUI-qwen21 over the /prompt API for Qwen-Image-2.1 t2i / edit.

Builds API-format graphs mirroring the official templates
(image_qwen_image_2_1_t2i.json / image_qwen_image_2_1_image_edit.json, prompt
enhancer branch skipped), with either UNETLoader (official) or SGLDUNETLoader
(integrated). Each run records wall time from POST to history completion.
"""

import argparse
import json
import os
import shutil
import time
import urllib.request
import uuid

T2I_PROMPT = (
    "Greyscale fashion editorial portrait of an avant-garde woman tilted upwards in "
    "profile, captured with striking high-contrast chiaroscuro lighting. She wears "
    "oversized, thick-rimmed circular black sunglasses with matte dark lenses, "
    "concealing her upward gaze. The striking background is a vivid mixed-media "
    "graphic collage, composed of flat vibrant lime green planes, stark black and "
    "white overlapping circles, segmented semi-circles with bold zebra striping, "
    "dense micro-halftone dot matrices, and vector topographic contour lines. "
    "Absolutely no text, pure imagery only."
)
EDIT_PROMPT = "Turn the scene into a watercolor painting and make the background warm orange."


def build_graph(args, mode):
    unet = args.unet
    if mode == "official":
        loader = {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        }
    else:
        loader = {
            "class_type": "SGLDUNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        }
        if args.sgld_options:
            loader["inputs"]["sgld_options"] = ["20", 0]
    g = {
        "1": loader,
        "2": {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": ["1", 0], "device": args.cache_device, "dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"},
        },
        "5": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["3", 0],
                "prompt": args.prompt or (EDIT_PROMPT if args.task == "edit" else T2I_PROMPT),
                "negative_prompt": args.negative,
                "resolution": args.resolution,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": args.seed,
                "steps": args.steps,
                "cfg": args.cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": ["6", 0],
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 0]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": f"qi21_{args.task}_{mode}"},
        },
    }
    if args.task == "edit":
        g["5"]["inputs"]["vae"] = ["4", 0]
        for i, name in enumerate(args.images, start=1):
            nid = str(30 + i)
            g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
            g["5"]["inputs"][f"images.image_{i}"] = [nid, 0]
        g["7"]["inputs"]["latent_image"] = ["5", 2]
        if args.edit_target:
            g["6"] = {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": args.width, "height": args.height, "batch_size": 1},
            }
            g["7"]["inputs"]["latent_image"] = ["6", 0]
    else:
        g["6"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": args.width, "height": args.height, "batch_size": args.batch},
        }
    if mode != "official" and args.sgld_options:
        g["20"] = {"class_type": "SGLDOptions", "inputs": json.loads(args.sgld_options)}
    return g


def post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def run_once(args, mode, tag):
    base = f"http://127.0.0.1:{args.port}"
    graph = build_graph(args, mode)
    t0 = time.time()
    resp = post(base + "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
    pid = resp["prompt_id"]
    while True:
        hist = get(f"{base}/history/{pid}")
        if pid in hist:
            h = hist[pid]
            status = h.get("status", {})
            if status.get("completed") or status.get("status_str") in ("success", "error"):
                break
        time.sleep(0.2)
    dt = time.time() - t0
    if status.get("status_str") != "success":
        print(json.dumps(status, indent=1)[:4000])
        raise SystemExit(f"run failed: {tag}")
    img = h["outputs"]["9"]["images"][0]
    src = os.path.join(args.comfy_dir, "output", img.get("subfolder", ""), img["filename"])
    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, f"{tag}.png")
    shutil.copy(src, dst)
    os.remove(src)
    print(json.dumps({"tag": tag, "mode": mode, "seconds": round(dt, 3), "image": dst}))
    return dt, dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["official", "integrated"], required=True)
    p.add_argument("--task", choices=["t2i", "edit"], default="t2i")
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--unet", default="qwen_image_2.1_bf16.safetensors")
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative", default="")
    p.add_argument("--seed", type=int, default=447606998181262)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--cache-device", default="auto")
    p.add_argument("--sgld-options", default=None, help="JSON inputs for SGLDOptions")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--edit-target", action="store_true", help="edit with an EmptyLatentImage of --width x --height")
    p.add_argument("--tag", default=None)
    p.add_argument("--comfy-dir", default="/scratch/data/sgld_comfy/ComfyUI-qwen21")
    p.add_argument("--out-dir", default="/scratch/data/sgld_comfy/results/qwen_image21/images")
    args = p.parse_args()
    times = []
    base_seed = args.seed
    for i in range(args.runs):
        # ComfyUI caches identical graphs; a new seed per run forces the
        # sampler to execute. Official and integrated use the same seed list.
        args.seed = base_seed + i
        tag = f"{args.tag or args.task + '_' + args.mode}_s{args.seed}"
        dt, _ = run_once(args, args.mode, tag)
        times.append(dt)
    print(json.dumps({"mode": args.mode, "task": args.task, "times": [round(t, 3) for t in times]}))


if __name__ == "__main__":
    main()
