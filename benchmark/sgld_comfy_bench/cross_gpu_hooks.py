"""Opt-in same-input A/B diagnostic. Never used for normal workflow timing."""
import functools
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def tensors(value, path=""):
    import torch
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, part in value.items():
            yield from tensors(part, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for key, part in enumerate(value):
            yield from tensors(part, f"{path}/{key}")


def fingerprint(value):
    return {p: hashlib.sha256(t.detach().contiguous().cpu().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
            for p, t in tensors(value)}


def cpu_outputs(value):
    return {p: t.detach().cpu().clone() for p, t in tensors(value)}


def difference(a, b):
    return {p: {"exact": bool(a[p].equal(b[p])),
                "max_abs": float((a[p].float() - b[p].float()).abs().max()) if a[p].numel() else 0.,
                "rms": float((a[p].float() - b[p].float()).square().mean().sqrt()) if a[p].numel() else 0.}
            for p in a}


def install(cls, kind):
    if not os.getenv("CROSS_DIAGNOSTIC_DIR"):
        return
    original = cls.forward
    calls = 0

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal calls
        import torch
        calls += 1
        if calls != 3:  # Two full warmup steps precede the captured input.
            return original(self, *args, **kwargs)
        directory = Path(os.environ["CROSS_DIAGNOSTIC_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        inputs = (args, kwargs)
        # Qwen fills the caller's initially empty prefix-cache dictionaries on
        # the first step of a generation. Complete that normal operation once
        # before fingerprinting the steady-step input, and retain its output.
        request_output = original(self, *args, **kwargs)
        before = fingerprint(inputs)
        methods = sorted({type(m.quant_method).__name__ for m in self.modules() if hasattr(m, "quant_method")})
        report = {"kind": kind, "quant_methods": methods, "inputs": [
            {"path": p, "shape": list(t.shape), "dtype": str(t.dtype)} for p, t in tensors(inputs)],
            "timing": "synchronized model forward; no profiler; diagnostic workflow time is invalid",
            "comparisons": []}
        mlps = [m for m in self.modules() if type(m).__name__ == "QwenImage21FeedForward"]
        packed = [(m, getattr(m, "_int8_gate_up_packed", False), getattr(m, "_bf16_gate_up_packed", False)) for m in mlps]
        report["packed_int8_count"] = sum(a for _, a, _ in packed)
        report["packed_bf16_count"] = sum(b for _, _, b in packed)
        from sglang.multimodal_gen.runtime.layers.quantization import convrot_int8_comfy_kitchen as kitchen
        old_rows = kitchen._MAX_ROWS_PER_CALL
        old_fusion = os.environ.get("SGLANG_H3_FUSED_FP32_MODULATION")

        def set_variant(test, variant):
            if kind == "h3":
                kitchen._MAX_ROWS_PER_CALL = (8192 if variant == "A" else 0) if test == "row_split" else 0
                os.environ["SGLANG_H3_FUSED_FP32_MODULATION"] = "1" if test == "fp32_modulation" and variant == "B" else "0"
            else:
                for m, a, b in packed:
                    m._int8_gate_up_packed = a and variant == "B"
                    m._bf16_gate_up_packed = b and variant == "B"

        def forward():
            return original(self, *args, **kwargs)

        try:
            for test in (["row_split", "fp32_modulation"] if kind == "h3" else ["qwen_mlp"]):
                # Prime both paths before collecting any timing or reference.
                for variant in "AB":
                    set_variant(test, variant)
                    forward()
                    torch.cuda.synchronize()
                samples = []
                outputs = {}
                repeat_diffs = []
                for i, variant in enumerate("ABBAABBA"):
                    set_variant(test, variant)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = forward()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    current = cpu_outputs(result)
                    if variant in outputs:
                        repeat_diffs.append({"variant": variant, **difference(outputs[variant], current)})
                    outputs[variant] = current
                    samples.append({"index": i, "variant": variant, "seconds": seconds})
                    del result
                medians = {v: statistics.median(r["seconds"] for r in samples if r["variant"] == v) for v in "AB"}
                comparison = {"test": test, "samples": samples, "medians_s": medians,
                              "benefit_percent": 100 * (1 - medians["B"] / medians["A"]),
                              "output_difference": difference(outputs["A"], outputs["B"]),
                              "repeat_differences": repeat_diffs}
                if kind == "h3":
                    # Count the actual new kernel calls on an extra untimed pass.
                    import sglang.multimodal_gen.runtime.models.dits.minimax_h3 as h3
                    counts = {}
                    originals = {}
                    names = ["indexed_gate_fp32", "indexed_scale_shift_fp32"]
                    for name in names:
                        fn = getattr(h3, name)
                        originals[name] = fn
                        def counted(*a, _fn=fn, _name=name, **kw):
                            counts[_name] = counts.get(_name, 0) + 1
                            return _fn(*a, **kw)
                        setattr(h3, name, counted)
                    split_shapes = {}
                    row_fn = kitchen._row_split
                    def count_split(rows, out_features):
                        value = row_fn(rows, out_features)
                        key = f"{rows}x{out_features}:split={value}"
                        split_shapes[key] = split_shapes.get(key, 0) + 1
                        return value
                    try:
                        set_variant(test, "B")
                        kitchen._row_split = count_split
                        forward()
                        torch.cuda.synchronize()
                    finally:
                        kitchen._row_split = row_fn
                        for name, fn in originals.items():
                            setattr(h3, name, fn)
                    comparison.update(candidate_kernel_calls=counts, candidate_linear_shapes=split_shapes)
                report["comparisons"].append(comparison)
                report["inputs_unchanged"] = before == fingerprint(inputs)
                (directory / "ablation.json").write_text(json.dumps(report, indent=2))
                if not report["inputs_unchanged"]:
                    raise RuntimeError("same-input replay mutated an input; reject diagnostic")
        finally:
            kitchen._MAX_ROWS_PER_CALL = old_rows
            if old_fusion is None:
                os.environ.pop("SGLANG_H3_FUSED_FP32_MODULATION", None)
            else:
                os.environ["SGLANG_H3_FUSED_FP32_MODULATION"] = old_fusion
            for m, a, b in packed:
                m._int8_gate_up_packed, m._bf16_gate_up_packed = a, b
        return request_output

    cls.forward = wrapped
