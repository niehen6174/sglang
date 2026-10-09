"""Compare two ComfyUI output dirs (video mp4 + saved AV latents): PSNR / SSIM-lite / latent stats.

usage: compare_outputs.py DIR_A DIR_B [--json out.json]
"""

import argparse
import glob
import json
import os

import av
import numpy as np
import torch
from safetensors.torch import load_file


def frames_and_audio(path):
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    audio = []
    container = av.open(path)
    if container.streams.audio:
        for f in container.decode(audio=0):
            audio.append(f.to_ndarray().astype(np.float32))
    container.close()
    audio = np.concatenate(audio, axis=-1) if audio else None
    return np.stack(frames).astype(np.float64), audio


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def latent(path):
    return load_file(path)["latent_tensor"].float()


def tensor_stats(a, b):
    a, b = a.flatten(), b.flatten()
    diff = (a - b).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_l2": (diff.norm() / a.norm()).item(),
        "cos": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    args = ap.parse_args()
    va, aa = frames_and_audio(glob.glob(os.path.join(args.a, "*.mp4"))[0])
    vb, ab = frames_and_audio(glob.glob(os.path.join(args.b, "*.mp4"))[0])
    per_frame = [psnr(x, y) for x, y in zip(va, vb)]
    out = {
        "a": args.a,
        "b": args.b,
        "frames": [len(va), len(vb)],
        "size": list(va.shape[1:3]),
        "video_psnr_mean": float(np.mean(per_frame)),
        "video_psnr_min": float(np.min(per_frame)),
        "video_psnr_all_frames": psnr(va, vb),
        "video_mean_abs_8bit": float(np.mean(np.abs(va - vb))),
    }
    if aa is not None and ab is not None:
        n = min(aa.shape[-1], ab.shape[-1])
        x, y = aa[..., :n], ab[..., :n]
        out["audio_snr_db"] = float(10 * np.log10(np.sum(x**2) / max(np.sum((x - y) ** 2), 1e-20)))
        out["audio_corr"] = float(np.corrcoef(x.flatten(), y.flatten())[0, 1])
    for kind in ("video_latent", "audio_latent"):
        fa = glob.glob(os.path.join(args.a, f"{kind}_*.latent"))
        fb = glob.glob(os.path.join(args.b, f"{kind}_*.latent"))
        if fa and fb:
            out[kind] = tensor_stats(latent(fa[0]), latent(fb[0]))
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
