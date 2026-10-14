"""Isolated cross-GPU ablations using one pinned fork commit and cached models.

Run under the host's original kit env.sh, then pass card and original source.
No shared ComfyUI custom-node links, models, git branches or GPU clocks change.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("card", choices=("4090", "5090", "h200"))
    parser.add_argument("--original-source", required=True, type=Path)
    parser.add_argument("--stage", choices=("diagnostic", "formal"), default="diagnostic")
    parser.add_argument("--cases", default="")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    card = args.card
    source = Path(__file__).resolve().parents[2]
    kit_source = Path(__file__).resolve().parent
    root = source.parent
    gpu = "1" if card == "h200" else "0"
    assert not command("nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"), "GPU occupied"
    lock = open(root / "experiment.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original = args.original_source
    comfy_source = Path(os.environ["COMFY_DIR"])
    base = root / "comfy-base"
    if not base.exists():
        (base / "custom_nodes").mkdir(parents=True)
        for name in ("models", "input"):
            (base / name).symlink_to(comfy_source / name)
        (base / "output").mkdir()
    runtime = root / ("runtime-diagnostic" if args.stage == "diagnostic" else "source")
    if args.stage == "diagnostic" and not runtime.exists():
        shutil.copytree(source / "python/sglang", runtime / "python/sglang", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(kit_source / "cross_gpu_hooks.py", runtime / "python/cross_gpu_hooks.py")
        for file, cls, kind in (("minimax_h3.py", "MiniMaxH3DiTModel", "h3"), ("qwen_image21.py", "QwenImage21Transformer2DModel", "qwen")):
            path = runtime / "python/sglang/multimodal_gen/runtime/models/dits" / file
            with path.open("a") as f:
                f.write(f"\nfrom cross_gpu_hooks import install as _cross_install\n_cross_install({cls}, {kind!r})\n")
    for name, target in {"ComfyUI_SGLDiffusion": runtime / "python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion",
                         "H3Benchmark": kit_source / "custom_nodes/H3Benchmark"}.items():
        p = base / "custom_nodes" / name
        if p.is_symlink():
            p.unlink()
        p.symlink_to(target)
    private_kit = root / "launch-kit"
    (private_kit / "scripts").mkdir(parents=True, exist_ok=True)
    # Source credentials without copying them. Override inherited read-only cache paths.
    (private_kit / "env.sh").write_text(
        f'source {original}/benchmark/sgld_comfy_bench/env.sh || true\n'
        f'export KIT_ROOT={private_kit} SGLANG_SRC={runtime} PYTHONPATH={runtime}/python\n'
        f'export HF_HUB_CACHE={os.environ["WORKSPACE"]}/cache/huggingface/hub\n')
    launcher = (kit_source / "scripts/start-comfyui.sh").read_text()
    launcher = launcher.replace('exec python main.py', 'exec "${CROSS_GPU_PYTHON:?}" main.py')
    launcher = launcher.replace('${COMFY_EXTRA_ARGS:-}', f'--base-directory {base} --output-directory {base}/output ${{COMFY_EXTRA_ARGS:-}}')
    (private_kit / "scripts/start-comfyui.sh").write_text(launcher)
    (private_kit / "scripts/start-comfyui.sh").chmod(0o755)
    os.environ.update(KIT_ROOT=str(private_kit), SGLANG_SRC=str(runtime), PYTHONPATH=str(runtime / "python"),
                      GPUS=gpu, LANES="1", PORT_BASE="8790", CROSS_GPU_PYTHON=os.environ["CROSS_GPU_PYTHON"],
                      H3_BENCH_SAMPLE_HOST_MEMORY="0")
    for key in ("SGLANG_H3_FUSED_FP32_GATE", "SGLANG_H3_FUSED_FP32_MODULATION", "SGLANG_KITCHEN_INT8_MAX_ROWS",
                "SGLANG_QWEN21_FUSED_INT8_MLP", "SGLANG_QWEN21_PACKED_BF16_MLP", "SGLANG_DIFFUSION_CONVROT_INT8_BACKEND"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(kit_source / "bench"))
    import comfyserver as comfy
    import experiments_a as experiments
    import graphs
    comfy.COMFY = base
    identity = {"card": card, "gpu": gpu, "commit": command("git", "-C", str(source), "rev-parse", "HEAD"),
                "original_commit": command("git", "-C", str(original), "rev-parse", "HEAD"),
                "clock_policy": "unchanged", "host_memory_sampling": False,
                "gpu_identity": command("nvidia-smi", "-i", gpu, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"),
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "hooks_sha256": hashlib.sha256((kit_source / "cross_gpu_hooks.py").read_bytes()).hexdigest()}
    (root / f"identity-{args.stage}.json").write_text(json.dumps(identity, indent=2))
    cases = [("h3-480", "h3", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "comfy_kitchen", 0),
             ("h3-768", "h3", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "comfy_kitchen", 0),
             ("h3-ref4", "h3", "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "comfy_kitchen", 4),
             ("qwen-int8-cfg1", "qwen", "qwen_image_2.1_int8_convrot.safetensors", "comfy_kitchen", 1),
             ("qwen-int8-cfg4", "qwen", "qwen_image_2.1_int8_convrot.safetensors", "comfy_kitchen", 4),
             ("qwen-bf16-cfg1", "qwen", "qwen_image_2.1_bf16.safetensors", "", 1),
             ("qwen-bf16-cfg4", "qwen", "qwen_image_2.1_bf16.safetensors", "", 4)]
    if card == "h200":
        cases = [("h3-bf16-default", "h3", "minimax_h3_fl2va_bf16.safetensors", "", 0),
                 ("h3-int8-default", "h3", "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "", 0),
                 ("qwen-int8-default", "qwen", "qwen_image_2.1_int8_convrot.safetensors", "", 1)] + cases
    wanted = set(args.cases.split(",")) if args.cases else None
    if wanted:
        assert wanted <= {c[0] for c in cases}, "unknown case"
        cases = [c for c in cases if c[0] in wanted]
    def interrupt(signum, _):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupt)
    for name, model, unet, quant_backend, cfg_refs in cases:
        variants = ["ablation"] if args.stage == "diagnostic" else (["rows8192", "rows0", "fused"] if model == "h3" else ["old", "packed"])
        for variant in variants:
            target = root / args.stage / name / variant
            if (target / "complete.json").exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            attention = "fa" if card != "5090" else ("torch_sdpa" if cfg_refs == 4 and model == "h3" else "torch_cudnn_sdpa")
            exp = experiments._exp(name, model, "sgld", attn=attention,
                                  **({"canvas": experiments.H3_480 if name == "h3-480" else experiments.H3_768, "refs": cfg_refs} if model == "h3" else {"cfg": cfg_refs}))
            resolved = experiments.resolve(exp, card, target)
            for node in resolved["graph"].values():
                if node["class_type"] == "SGLDUNETLoader":
                    node["inputs"]["unet_name"] = unet
            env = {"SGLANG_KITCHEN_INT8_MAX_ROWS": "8192" if variant == "rows8192" else "0",
                   "SGLANG_H3_FUSED_FP32_MODULATION": "1" if variant == "fused" else "0",
                   "SGLANG_QWEN21_FUSED_INT8_MLP": "1" if variant in ("packed", "ablation") else "0",
                   "SGLANG_QWEN21_PACKED_BF16_MLP": "1" if variant in ("packed", "ablation") else "0"}
            if quant_backend:
                env["SGLANG_DIFFUSION_CONVROT_INT8_BACKEND"] = quant_backend
            if args.stage == "diagnostic":
                env["CROSS_DIAGNOSTIC_DIR"] = str(target / "direct")
            settings = {"name": name, "variant": variant, "stage": args.stage, "unet": unet,
                        "attention": attention, "quant_backend_request": quant_backend or "default",
                        "env": env, "sgld_options": resolved["sgld_options"]}
            (target / "config.json").write_text(json.dumps(settings, indent=2))
            print(json.dumps({**settings, "status": "starting"}), flush=True)
            server, monitor = comfy.Server("gpu0", target), comfy.make_monitor("gpu0")
            rows = []
            try:
                server.start(resolved["flags"], {**resolved["env"], **env}, monitor=monitor)
                phases = [("warm", 149998, 2)]
                phases += [("diagnostic_00", 150000, 2)] if args.stage == "diagnostic" else [(f"m{i}_00", 150000+i, args.steps) for i in range(args.repetitions)]
                for phase, seed, steps in phases:
                    graph = graphs.set_seed(graphs.with_steps(resolved["graph"], model, steps), seed)
                    result = comfy.run_prompt(server, graph, name + "__" + phase, monitor, 2400, keep_media=True)
                    row = {**settings, "phase": phase, "seed": seed, "steps": steps, "result": result}
                    if result["status"] == "success":
                        calls = result["profile"]["apply_model_seconds"]
                        row.update(denoise_s=sum(result["profile"]["sampler_seconds"]), steady_step_median_s=statistics.median(calls[1:] or calls))
                    with (root / "results.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps({k: v for k, v in row.items() if k != "result"} | {"status": result["status"], "e2e_s": result.get("e2e_s"), "error": result.get("error", "")[:600]}), flush=True)
                    rows.append(row)
                    if result["status"] != "success":
                        break
                if all(r["result"]["status"] == "success" for r in rows):
                    if args.stage == "diagnostic":
                        assert (target / "direct/ablation.json").exists(), "diagnostic hook did not run"
                    (target / "complete.json").write_text(json.dumps({"finished_unix": time.time()}))
            finally:
                server.stop()
                monitor.shutdown()
            if not all(r["result"]["status"] == "success" for r in rows):
                print(json.dumps({"case_failed": name, "variant": variant}), flush=True)
    (root / f"finished-{args.stage}.json").write_text(json.dumps({"finished_unix": time.time(), "cases": [c[0] for c in cases]}))


if __name__ == "__main__":
    main()
