"""Local performance measurement using ComfyUI's model-patcher wrappers."""

import hashlib
import json
import pathlib
import os
import time

import torch
from safetensors.torch import save_file
from comfy.patcher_extension import WrappersMP

ROOT = pathlib.Path(os.environ["H3_BENCH_SUITE"]).resolve() / "runs"


def directory(run_id):
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in run_id
    ):
        raise ValueError("Invalid benchmark run id")
    path = ROOT / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def capture_inputs(args, kwargs, path):
    tensors = {}

    def visit(value, name):
        if isinstance(value, torch.Tensor):
            if getattr(value, "is_nested", False):
                return [
                    visit(part, name + "/" + str(i))
                    for i, part in enumerate(value.unbind())
                ]
            cpu = value.detach().contiguous().cpu()
            digest = hashlib.sha256(
                cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            tensors[name] = cpu.clone()
            return {
                "tensor": name,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "sha256": digest,
            }
        if isinstance(value, dict):
            return {
                str(k): visit(v, name + "/" + str(k))
                for k, v in value.items()
                if str(k) not in {"patches", "patches_replace", "wrappers", "callbacks"}
            }
        if isinstance(value, (list, tuple)):
            return [visit(v, name + "/" + str(i)) for i, v in enumerate(value)]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if type(value).__name__ == "PackedLayout":
            return visit(vars(value), name + "/layout")
        return {"type": type(value).__name__}

    metadata = {"args": visit(args, "args"), "kwargs": visit(kwargs, "kwargs")}
    save_file(tensors, str(path / "first-apply-inputs.safetensors"))
    (path / "first-apply-inputs.json").write_text(json.dumps(metadata, indent=2))


class H3BenchmarkModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",), "run_id": ("STRING",)},
            "optional": {"capture_first_inputs": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "bind"
    CATEGORY = "H3 Benchmark"

    def bind(self, model, run_id, capture_first_inputs=False):
        path = directory(run_id)
        clone = model.clone()
        stats = {"apply_model_seconds": [], "sampler_seconds": []}

        def forward(executor, *args, **kwargs):
            if capture_first_inputs and not stats["apply_model_seconds"]:
                capture_inputs(args, kwargs, path)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = executor(*args, **kwargs)
            torch.cuda.synchronize()
            stats["apply_model_seconds"].append(time.perf_counter() - start)
            return out

        def sample(executor, *args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                result = executor(*args, **kwargs)
                torch.cuda.synchronize()
                return result
            finally:
                stats["sampler_seconds"].append(time.perf_counter() - start)
                (path / "sampler-profile.json").write_text(json.dumps(stats, indent=2))

        clone.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "h3_benchmark", forward)
        clone.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, "h3_benchmark", sample)
        return (clone,)


class H3BenchmarkLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "run_id": ("STRING",)}}

    RETURN_TYPES = ("LATENT",)
    OUTPUT_NODE = True
    FUNCTION = "record"
    CATEGORY = "H3 Benchmark"

    def record(self, latent, run_id):
        samples = latent["samples"]
        parts = samples.unbind() if getattr(samples, "is_nested", False) else [samples]
        info = {}
        saved = {}
        for name, tensor in zip(["video", "audio"], parts):
            value = tensor.detach().contiguous().cpu()
            saved[name] = value
            info[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(torch.isfinite(value).all()),
                "sha256": hashlib.sha256(
                    value.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        path = directory(run_id)
        if run_id.endswith("_00"):
            save_file(saved, str(path / "latents.safetensors"))
        (path / "latent-info.json").write_text(json.dumps(info, indent=2))
        return (latent,)


class H3BenchmarkAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["pytorch", "sage", "sage3", "comfy_kitchen_int8"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "bind"
    CATEGORY = "H3 Benchmark"

    def bind(self, model, backend):
        from comfy.ldm.modules.attention import get_attention_function

        function = get_attention_function(backend, None)
        if function is None:
            raise ValueError(f"Official attention backend unavailable: {backend}")
        clone = model.clone()
        clone.set_model_optimized_attention(function)
        return (clone,)


NODE_CLASS_MAPPINGS = {
    "H3BenchmarkAttention": H3BenchmarkAttention,
    "H3BenchmarkModel": H3BenchmarkModel,
    "H3BenchmarkLatent": H3BenchmarkLatent,
}
