"""Quality of each config against its reference on the same card and engine (EXPERIMENTS.md 7.3).

For every pair (variant, reference) whose first measured output exists: PSNR and SSIM over 8
evenly spaced decoded frames (luma), and the latent cosine when both latents were saved (track A).
Distilled models (FastH3 / VDN) are not compared pixelwise; they only go into the frame sheets.
Writes SUITE/quality.jsonl and SUITE/sheets/<group>.jpg (3 frames per config, one row each).

Usage: quality.py --suite SUITE   (SUITE is the card directory holding A/ and B/)
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageDraw

N_FRAMES = 8


def _a_pairs():
    P = []
    for s in ("", "-480"):
        P += [(f"A-H3-10{s}", f"A-H3-01{s}", "parity"), (f"A-H3-11{s}", f"A-H3-01{s}", "parity"),
              (f"A-H3-27{s}", f"A-H3-11{s}", "lossy"), (f"A-H3-30{s}", f"A-H3-11{s}", "parallel")]
    P += [("A-H3-02", "A-H3-01", "lossy"), ("A-H3-03", "A-H3-01", "lossy"), ("A-H3-04", "A-H3-01", "lossy"),
          ("A-H3-20", "A-H3-11", "lossy"), ("A-H3-21", "A-H3-11", "lossy"), ("A-H3-22", "A-H3-11", "lossy"),
          ("A-H3-23", "A-H3-11", "lossy"), ("A-H3-24", "A-H3-23", "kernel"), ("A-H3-31", "A-H3-11", "parallel"),
          ("A-H3-32", "A-H3-27", "parallel"), ("A-H3-41", "A-H3-40", "parity"), ("A-H3-42", "A-H3-41", "lossy"),
          ("A-H3-43", "A-H3-41", "parallel"),
          ("A-LTX-10", "A-LTX-01", "parity"), ("A-LTX-11", "A-LTX-01", "parity"), ("A-LTX-02", "A-LTX-01", "lossy"),
          ("A-LTX-20", "A-LTX-11", "lossy"), ("A-LTX-21", "A-LTX-11", "lossy"), ("A-LTX-11", "A-LTX-23", "lossy"),
          ("A-LTX-30", "A-LTX-11", "parallel"), ("A-LTX-31", "A-LTX-20", "parallel")]
    for c in ("-cfg1", "-cfg4"):
        P += [(f"A-QW-10{c}", f"A-QW-01{c}", "parity"), (f"A-QW-11{c}", f"A-QW-01{c}", "parity"),
              (f"A-QW-02{c}", f"A-QW-01{c}", "lossy"), (f"A-QW-20{c}", f"A-QW-11{c}", "lossy"),
              (f"A-QW-21{c}", f"A-QW-11{c}", "lossy"), (f"A-QW-23{c}", f"A-QW-11{c}", "lossy"),
              (f"A-QW-30{c}", f"A-QW-11{c}", "parallel"), (f"A-QW-31{c}", f"A-QW-11{c}", "parallel")]
    return P


A_SHEETS = {
    "A-H3-768": ["A-H3-01", "A-H3-11", "A-H3-20", "A-H3-22", "A-H3-23", "A-H3-27", "A-H3-25", "A-H3-26", "A-H3-28",
                 "A-H3-30"],
    "A-H3-ref4": ["A-H3-40", "A-H3-41", "A-H3-42", "A-H3-43"],
    "A-LTX": ["A-LTX-01", "A-LTX-11", "A-LTX-20", "A-LTX-23", "A-LTX-30"],
    "A-QW-cfg1": ["A-QW-01-cfg1", "A-QW-11-cfg1", "A-QW-20-cfg1", "A-QW-23-cfg1", "A-QW-30-cfg1"],
    "A-QW-cfg4": ["A-QW-01-cfg4", "A-QW-11-cfg4", "A-QW-20-cfg4", "A-QW-23-cfg4", "A-QW-30-cfg4"],
}
B_PAIRS = [("B-H3-10", "B-H3-02", "lossy"), ("B-H3-11", "B-H3-02", "lossy"), ("B-H3-12", "B-H3-02", "lossy"),
           ("B-H3-13", "B-H3-02", "lossy"), ("B-H3-03", "B-H3-02", "parallel"), ("B-H3-04", "B-H3-02", "parallel"),
           ("B-LTX-10", "B-LTX-01", "lossy"), ("B-LTX-11", "B-LTX-01", "lossy"), ("B-LTX-02", "B-LTX-01", "parallel"),
           ("B-QW-10", "B-QW-01", "lossy"), ("B-QW-11", "B-QW-01", "lossy"), ("B-QW-12", "B-QW-01", "lossy"),
           ("B-QW-04", "B-QW-01", "parallel"), ("B-QW-03", "B-QW-02", "parallel")]
B_SHEETS = {"B-H3": ["B-H3-02", "B-H3-10", "B-H3-11", "B-H3-12", "B-H3-13", "B-H3-14"],
            "B-H3-ref8": ["B-H3-30", "B-H3-31"], "B-LTX": ["B-LTX-01", "B-LTX-03", "B-LTX-10", "B-LTX-11"],
            "B-QW": ["B-QW-01", "B-QW-02", "B-QW-10", "B-QW-11", "B-QW-12"]}


def media_a(suite, eid):
    d = suite / "A" / "runs" / f"{eid}__m0_00"
    return next((p for p in sorted(d.glob("*")) if p.suffix in (".mp4", ".png")), None) if d.exists() else None


def media_b(suite, eid, engine):
    d = suite / "B" / "runs" / f"{eid}@{engine}__m0"
    return next((p for p in (d / "output.mp4", d / "output.png") if p.exists()), None)


def frames(path, n=N_FRAMES):
    """n evenly spaced RGB frames as uint8 arrays (an image is one frame)."""
    if path.suffix == ".png":
        return [np.asarray(Image.open(path).convert("RGB"))]
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                            "stream=nb_read_frames,width,height", "-of", "json", str(path)], capture_output=True, text=True)
    st = json.loads(probe.stdout)["streams"][0]
    total, w, h = int(st["nb_read_frames"]), int(st["width"]), int(st["height"])
    idx = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='{sel}'", "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return list(arr)


def _luma(x):
    x = x.astype(np.float64)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def psnr(a, b):
    mse = np.mean((_luma(a) - _luma(b)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _blur(x, sigma=1.5, radius=5):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    x = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, x)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, x)


def ssim(a, b):
    x, y = _luma(a), _luma(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _blur(x), _blur(y)
    sxx, syy, sxy = _blur(x * x) - mx ** 2, _blur(y * y) - my ** 2, _blur(x * y) - mx * my
    return float(np.mean(((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))))


def latent_cos(suite, var, ref):
    from safetensors.torch import load_file

    pa, pb = (suite / "A" / "runs" / f"{x}__m0_00" / "latents.safetensors" for x in (var, ref))
    if not (pa.exists() and pb.exists()):
        return None
    la, lb = load_file(str(pa)), load_file(str(pb))
    out = {}
    for k in la:
        if k in lb and la[k].shape == lb[k].shape:
            a, b = la[k].double().flatten(), lb[k].double().flatten()
            out[k] = round(float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)), 5)
    return out


def compare(pa, pb):
    fa, fb = frames(pa), frames(pb)
    if len(fa) != len(fb) or fa[0].shape != fb[0].shape:
        return {"error": f"shape mismatch {len(fa)}x{fa[0].shape} vs {len(fb)}x{fb[0].shape}"}
    return {"psnr_db": round(float(np.mean([psnr(a, b) for a, b in zip(fa, fb)])), 2),
            "ssim": round(float(np.mean([ssim(a, b) for a, b in zip(fa, fb)])), 4)}


def sheet(rows, out, label_w=220, thumb_h=160):
    """rows: [(label, media path)] -> one JPEG, 3 frames per row."""
    tiles = []
    for label, path in rows:
        if path is None:
            continue
        fr = frames(path, 3) if path.suffix == ".mp4" else frames(path)
        ims = [Image.fromarray(f).resize((int(f.shape[1] * thumb_h / f.shape[0]), thumb_h)) for f in fr]
        tiles.append((label, ims))
    if not tiles:
        return
    width = label_w + max(sum(i.width for i in ims) for _, ims in tiles)
    canvas = Image.new("RGB", (width, thumb_h * len(tiles)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (label, ims) in enumerate(tiles):
        draw.text((8, r * thumb_h + thumb_h // 2 - 6), label, fill="black")
        x = label_w
        for im in ims:
            canvas.paste(im, (x, r * thumb_h))
            x += im.width
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    a = ap.parse_args()
    suite = pathlib.Path(a.suite).resolve()
    rows = []
    for var, ref, kind in _a_pairs():
        pv, pr = media_a(suite, var), media_a(suite, ref)
        if pv and pr:
            row = {"track": "A", "variant": var, "reference": ref, "kind": kind, **compare(pv, pr),
                   "latent_cos": latent_cos(suite, var, ref)}
            rows.append(row)
    for engine in ("sglang", "vllm"):
        for var, ref, kind in B_PAIRS:
            pv, pr = media_b(suite, var, engine), media_b(suite, ref, engine)
            if pv and pr:
                rows.append({"track": "B", "engine": engine, "variant": var, "reference": ref, "kind": kind, **compare(pv, pr)})
    for var in ("B-H3-01", "B-H3-02", "B-H3-30", "B-LTX-01", "B-LTX-03", "B-QW-01", "B-QW-02"):
        pv, pr = media_b(suite, var, "vllm"), media_b(suite, var, "sglang")
        if pv and pr:
            rows.append({"track": "B", "engine": "vllm vs sglang", "variant": var, "reference": var, "kind": "cross-engine",
                         **compare(pv, pr)})
    with open(suite / "quality.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r))
    for name, ids in A_SHEETS.items():
        sheet([(i, media_a(suite, i)) for i in ids], suite / "sheets" / f"{name}.jpg")
    for name, ids in B_SHEETS.items():
        sheet([(f"{i} {e}", media_b(suite, i, e)) for i in ids for e in ("sglang", "vllm")], suite / "sheets" / f"{name}.jpg")
    print(f"QUALITY_DONE ({len(rows)} pairs) -> {suite / 'quality.jsonl'}, {suite / 'sheets'}")


if __name__ == "__main__":
    main()
