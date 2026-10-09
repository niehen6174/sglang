#!/usr/bin/env python3
"""PSNR / max-abs between pairs of PNGs (all channels, incl. alpha)."""
import json
import sys

import numpy as np
from PIL import Image


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


out = {}
for pair in sys.argv[1:]:
    a_path, b_path = pair.split(",")
    a, b = np.asarray(Image.open(a_path)), np.asarray(Image.open(b_path))
    assert a.shape == b.shape, (a.shape, b.shape)
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    out[pair] = {
        "shape": list(a.shape),
        "psnr_db": round(psnr(a, b), 2),
        "mean_abs": round(float(d.mean()), 3),
        "max_abs": int(d.max()),
        "pct_pixels_diff_gt8": round(float((d.max(-1) > 8).mean() * 100), 3),
    }
print(json.dumps(out, indent=1))
