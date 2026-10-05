"""Compare saved same-seed H3 latents; numerical agreement is not a quality rating."""

import argparse
import json
import pathlib
import os

import torch
from safetensors.torch import load_file

SUITE = pathlib.Path(os.environ["H3_BENCH_SUITE"]).resolve()


def compare(left, right):
    a = load_file(str(left), device="cpu")
    b = load_file(str(right), device="cpu")
    result = {}
    for name in a.keys() | b.keys():
        if name not in a or name not in b:
            result[name] = {"error": "component missing"}
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape:
            result[name] = {
                "error": "shape mismatch",
                "left_shape": list(x.shape),
                "right_shape": list(y.shape),
            }
            continue
        xf, yf = x.double().flatten(), y.double().flatten()
        finite = bool(torch.isfinite(xf).all() and torch.isfinite(yf).all())
        entry = {
            "shape": list(x.shape),
            "finite": finite,
            "bit_equal": x.dtype == y.dtype and torch.equal(x, y),
        }
        if finite:
            delta = xf - yf
            denom = xf.norm() * yf.norm()
            entry.update(
                cosine=float(xf.dot(yf) / denom) if denom else None,
                rmse=float(delta.square().mean().sqrt()),
                reference_rms=float(xf.square().mean().sqrt()),
                max_abs=float(delta.abs().max()),
            )
        result[name] = entry
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    left = SUITE / "runs" / (args.left + "_00")
    right = SUITE / "runs" / (args.right + "_00")
    workflows = []
    for directory in [left, right]:
        request = json.loads((directory / "workflow.json").read_text())
        workflows.append(request)
        sampler = next(
            v
            for v in request.values()
            if v.get("class_type") == "SamplerCustomAdvanced"
        )
        noise = request[str(sampler["inputs"]["noise"][0])]
        assert noise["inputs"]["noise_seed"] == 150000, noise
    comparable = {
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "LoadImage",
        "ManualSigmas",
        "KSamplerSelect",
        "RandomNoise",
        "VAELoader",
    }
    conditioning = [
        {
            key: node
            for key, node in workflow.items()
            if node.get("class_type") in comparable
        }
        for workflow in workflows
    ]
    assert (
        conditioning[0] == conditioning[1]
    ), "Conditioning or sampler mismatch: choose matching workflows"
    output = {
        "left": args.left,
        "right": args.right,
        "seed": 150000,
        "note": "Requires matching prompt, conditioning, weights and sampler; compare workflow.json. Approximate attention may alter latents. This does not establish perceptual quality.",
        "components": compare(
            left / "latents.safetensors", right / "latents.safetensors"
        ),
    }
    target = SUITE / "latent-comparisons"
    target.mkdir(exist_ok=True)
    (target / (args.left + "--" + args.right + ".json")).write_text(
        json.dumps(output, indent=2)
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
