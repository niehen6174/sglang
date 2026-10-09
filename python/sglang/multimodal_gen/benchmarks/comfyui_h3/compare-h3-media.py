"""Compare two decoded A/V files: per-frame RGB PSNR and audio SNR (same seed runs)."""
import json
import subprocess
import sys

import numpy as np


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    streams = json.loads(out)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    a = next(s for s in streams if s["codec_type"] == "audio")
    return int(v["width"]), int(v["height"]), int(a["sample_rate"]), int(a["channels"])


def video(path, w, h):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float64)


def audio(path, rate, ch):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(rate), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def main(left, right):
    meta = probe(left)
    assert meta == probe(right), (meta, probe(right))
    w, h, rate, ch = meta
    a, b = video(left, w, h), video(right, w, h)
    n = min(len(a), len(b))
    mse = ((a[:n] - b[:n]) ** 2).mean(axis=(1, 2, 3))
    psnr = np.where(mse == 0, np.inf, 10 * np.log10(255.0**2 / np.maximum(mse, 1e-12)))
    x, y = audio(left, rate, ch), audio(right, rate, ch)
    m = min(len(x), len(y))
    noise = ((x[:m] - y[:m]) ** 2).sum()
    snr = float("inf") if noise == 0 else 10 * np.log10((x[:m] ** 2).sum() / noise)
    result = {
        "frames": [len(a), len(b)],
        "video_identical": bool(n == len(a) == len(b) and (mse == 0).all()),
        "video_psnr_mean": float(np.mean(np.minimum(psnr, 100))),
        "video_psnr_min": float(np.min(psnr)),
        "audio_samples": [len(x), len(y)],
        "audio_identical": bool(m == len(x) == len(y) and noise == 0),
        "audio_snr_db": snr,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
