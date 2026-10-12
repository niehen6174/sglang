"""Record the machine and check what it can run, before any experiment.

Writes SUITE/machine.json: CPU, RAM, disks, GPUs + topology + power limits, driver/CUDA, package
versions in both venvs, commits, which attention backends / INT8 kernels SGLang resolves, and
which model files are present. Exit code 1 if a required model file or node is missing.

Usage: check_env.py --out DIR [--build-int8]
"""
import argparse
import importlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

import torch

import models

ATTENTION = ["fa", "torch_cudnn_sdpa", "torch_sdpa", "sage_attn", "sage_attn_3", "hybrid_window_attn_h3"]
PACKAGES = ["torch", "triton", "sglang", "sgl_kernel", "flash_attn", "sageattention", "sageattn3", "comfy_kitchen",
            "cache_dit", "diffusers", "transformers", "flashinfer"]


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def machine():
    mem_kb = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")))
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    disks = {}
    for d in ("/personal", "/scratch", os.environ["WORKSPACE"]):
        if os.path.isdir(d):
            u = shutil.disk_usage(d)
            disks[d] = {"free_gb": round(u.free / 1e9), "total_gb": round(u.total / 1e9)}
    return {"hostname": platform.node(), "cpu": cpu, "cpu_count": os.cpu_count(), "ram_gb": round(mem_kb / 1e6),
            "disks": disks, "kernel": platform.release(),
            "nvidia_smi": sh("nvidia-smi --query-gpu=index,name,memory.total,power.limit,clocks.max.sm,pci.bus_id,"
                             "driver_version --format=csv,noheader"),
            "topology": sh("nvidia-smi topo -m"),
            "gpu_processes": sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")}


def versions():
    out = {"python": platform.python_version(), "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "present")
        except Exception as e:  # noqa: BLE001
            out[name] = f"missing ({type(e).__name__})"
    vpy = pathlib.Path(os.environ.get("VENV_VLLM", "")) / "bin/python"
    out["vllm_omni_env"] = sh(f"{vpy} -c 'import vllm, vllm_omni, torch; print(vllm.__version__, torch.__version__)'") \
        if vpy.exists() else "missing"
    for key, repo in (("sglang_commit", "SGLANG_SRC"), ("comfyui_commit", "COMFY_DIR"), ("vllm_omni_commit", "VLLM_OMNI_SRC")):
        out[key] = sh(f"git -C {os.environ.get(repo, '/nonexistent')} log -1 --format='%h %s'")[:120]
    return out


def attention():
    from pkgutil import resolve_name

    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

    out = {}
    for name in ATTENTION:
        try:
            backend = AttentionBackendEnum[name.upper()]
            cls = current_platform.get_attn_backend_cls_str(backend, 128, torch.bfloat16)
            resolved = resolve_name(cls).get_enum().name.lower()
            out[name] = "ok" if resolved == name else f"falls back to {resolved}"
        except Exception as e:  # noqa: BLE001
            out[name] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
    return out


def quantization(build_int8):
    out = {}
    try:
        from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_jit import jit_convrot_unavailable_reason

        reason = jit_convrot_unavailable_reason()
        out["int8_jit"] = "ok" if reason is None else f"unavailable: {reason}"
        if build_int8 and reason is None:
            from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_jit import _load_jit_kernel

            _load_jit_kernel()
            out["int8_jit"] = "ok (kernel built)"
    except Exception as e:  # noqa: BLE001
        out["int8_jit"] = f"check failed: {type(e).__name__}: {str(e)[:160]}"
    try:
        from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_comfy_kitchen import comfy_kitchen_available

        out["int8_comfy_kitchen"] = "ok" if comfy_kitchen_available() else "unavailable"
    except Exception as e:  # noqa: BLE001
        out["int8_comfy_kitchen"] = f"check failed: {type(e).__name__}"
    return out


def model_files(card):
    root = pathlib.Path(os.environ["COMFY_DIR"]) / "models"
    out = {}
    for _, _, rel, sub in models.track_a_downloads(card):
        f = root / sub / pathlib.Path(rel).name
        out[f"{sub}/{f.name}"] = f"{f.stat().st_size / 1e9:.1f} GB" if f.exists() else "MISSING"
    for name, d in (("FastH3", models.FASTH3_DIR()), ("VDN", models.VDN_DIR())):
        out[name] = "ok" if (pathlib.Path(d) / "transformer").exists() else "MISSING (A-H3-25/26/28/33/34 will fail)"
    for name in ("ComfyUI_SGLDiffusion", "H3Benchmark"):
        out[f"custom_nodes/{name}"] = "ok" if (pathlib.Path(os.environ["COMFY_DIR"]) / "custom_nodes" / name).exists() else "MISSING"
    for i in range(1, 9):
        p = pathlib.Path(os.environ["COMFY_DIR"]) / "input" / f"ref_{i:02d}.jpg"
        out[f"input/{p.name}"] = "ok" if p.exists() else "MISSING"
    out["ffprobe"] = "ok" if shutil.which("ffprobe") else "MISSING (install ffmpeg)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-int8", action="store_true")
    a = ap.parse_args()
    card = os.environ["CARD"]
    report = {"card": card, "machine": machine(), "versions": versions(), "attention": attention(),
              "quantization": quantization(a.build_int8), "models": model_files(card)}
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "machine.json").write_text(json.dumps(report, indent=1))
    m = report["machine"]
    print(f"CARD={card}  host={m['hostname']}  cpu={m['cpu']} x{m['cpu_count']}  ram={m['ram_gb']} GB")
    print(m["nvidia_smi"])
    if m["gpu_processes"]:
        print(f"! GPU processes already running:\n{m['gpu_processes']}")
    for section in ("versions", "attention", "quantization", "models"):
        print(f"\n[{section}]")
        for k, v in report[section].items():
            bad = section != "versions" and not str(v).startswith(("ok", "present", "falls back")) and "GB" not in str(v)
            print(f"{'! ' if bad else '  '}{k:58s} {v}")
    missing = [k for k, v in report["models"].items() if str(v).startswith("MISSING") and k not in ("FastH3", "VDN")]
    print(f"\nWrote {out / 'machine.json'}")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
