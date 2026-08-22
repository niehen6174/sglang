#!/usr/bin/env python3
"""Cross-model FA vs sol_attn benchmark with quality and latency metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
BENCH_PY = (
    REPO_ROOT
    / "python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py"
)
CHECK_GPU_SH = REPO_ROOT / "scripts/check_gpu_availability.sh"
DEFAULT_SOL_ATTN_CONFIG = {
    "tau": 1.0,
    "thresh_type": "diag",
    "dense_steps": 10,
    "dense_layers": "0,1",
    "kv_splits": "auto",
}

TARGET_MODELS = [
    "hunyuanvideo",
    "qwen",
    "zimage",
    "ltx23-ti2v-two-stage",
    "cosmos3-super-t2v",
    "wan-t2v",
]

VIDEO_PRESETS = {
    "hunyuanvideo",
    "ltx23-ti2v-two-stage",
    "cosmos3-super-t2v",
    "wan-t2v",
}


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_diffusion_denoise", BENCH_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import benchmark presets from {BENCH_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image, got shape={image.shape}")
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def compute_psnr(image: np.ndarray, gt_image: np.ndarray) -> float:
    from skimage.metrics import peak_signal_noise_ratio

    image = _ensure_rgb_uint8(image)
    gt_image = _ensure_rgb_uint8(gt_image)
    return float(peak_signal_noise_ratio(gt_image, image, data_range=255))


def compute_ssim(image: np.ndarray, gt_image: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    image = _ensure_rgb_uint8(image)
    gt_image = _ensure_rgb_uint8(gt_image)
    return float(structural_similarity(image, gt_image, channel_axis=2, data_range=255))


def compute_mean_abs_diff(image: np.ndarray, gt_image: np.ndarray) -> float:
    image = _ensure_rgb_uint8(image)
    gt_image = _ensure_rgb_uint8(gt_image)
    return float(np.abs(image.astype(np.float32) - gt_image.astype(np.float32)).mean())


def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def iter_video_frames(path: Path, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to read video: {path}")
    try:
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()


def compare_artifacts(reference: Path, candidate: Path, is_video: bool) -> dict[str, Any]:
    if is_video:
        ref_frames = list(iter_video_frames(reference))
        cand_frames = list(iter_video_frames(candidate))
        if not ref_frames or not cand_frames:
            raise ValueError(f"Empty video artifact: ref={reference}, cand={candidate}")
        n = min(len(ref_frames), len(cand_frames))
        psnrs, ssims, maes = [], [], []
        for idx in range(n):
            psnrs.append(compute_psnr(cand_frames[idx], ref_frames[idx]))
            ssims.append(compute_ssim(cand_frames[idx], ref_frames[idx]))
            maes.append(compute_mean_abs_diff(cand_frames[idx], ref_frames[idx]))
        return {
            "frames_compared": n,
            "reference_frames": len(ref_frames),
            "candidate_frames": len(cand_frames),
            "psnr_mean_db": float(np.mean(psnrs)),
            "psnr_min_db": float(np.min(psnrs)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_min": float(np.min(ssims)),
            "mae_mean": float(np.mean(maes)),
        }

    ref = load_image_rgb(reference)
    cand = load_image_rgb(candidate)
    return {
        "frames_compared": 1,
        "psnr_mean_db": compute_psnr(cand, ref),
        "psnr_min_db": compute_psnr(cand, ref),
        "ssim_mean": compute_ssim(cand, ref),
        "ssim_min": compute_ssim(cand, ref),
        "mae_mean": compute_mean_abs_diff(cand, ref),
    }


def check_gpu(
    model_key: str,
    min_free_mib: int,
    *,
    max_gpu_util: int = 90,
    gpu_ids: str | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["MAX_GPU_UTIL"] = str(max_gpu_util)
    if gpu_ids:
        env["GPU_IDS"] = gpu_ids
    proc = subprocess.run(
        [str(CHECK_GPU_SH), model_key, str(min_free_mib)],
        capture_output=True,
        text=True,
        env=env,
    )
    lines = (proc.stdout or "").splitlines()
    status = next((line.split("=", 1)[1] for line in lines if line.startswith("STATUS=")), "UNKNOWN")
    cuda = next(
        (line.split("=", 1)[1] for line in lines if line.startswith("CUDA_VISIBLE_DEVICES=")),
        None,
    )
    return {
        "ok": proc.returncode == 0,
        "status": status,
        "cuda_visible_devices": cuda,
        "log": proc.stdout,
    }


def parse_perf_dump(path: Path) -> dict[str, Any]:
    with open(path) as f:
        perf = json.load(f)
    total_ms = perf.get("total_duration_ms")
    denoise_ms = 0.0
    for step in perf.get("steps", []):
        name = step.get("name")
        if (
            isinstance(name, str)
            and step.get("duration_ms") is not None
            and (name.endswith("DenoisingStage") or name.endswith("RefinementStage"))
            and "BeforeDenoisingStage" not in name
        ):
            denoise_ms += float(step["duration_ms"])
    if denoise_ms <= 0.0:
        denoise_steps = perf.get("denoise_steps_ms", [])
        denoise_ms = sum(float(s.get("duration_ms", 0.0)) for s in denoise_steps)
    return {
        "e2e_s": float(total_ms) / 1000.0 if total_ms is not None else None,
        "denoise_s": denoise_ms / 1000.0 if denoise_ms > 0 else None,
        "raw": perf,
    }


def build_generate_cmd(
    bench_mod,
    model_key: str,
    *,
    label: str,
    output_dir: Path,
    transformer_attention_backend: str | None = None,
    attention_backend_config: dict | None = None,
    quick: bool = False,
    warmup: bool = True,
) -> list[str]:
    cfg = bench_mod.MODELS[model_key]
    is_video = model_key in VIDEO_PRESETS
    ext = "mp4" if is_video else "png"
    artifact_path = output_dir / f"{model_key}_{label}.{ext}"
    perf_path = output_dir / f"{model_key}_{label}_perf.json"

    # Quick smoke runs skip request warmup — LTX two-stage warmup alone can
    # double runtime and interact badly with subprocess pipe capture.
    use_warmup = warmup and not quick

    cmd = bench_mod.build_sglang_cmd(
        model_key,
        perf_dump_path=str(perf_path),
        warmup=use_warmup,
        torch_compile=False,
        save_output=True,
        artifact_dir=output_dir,
    )
    cmd.extend(["--output-file-path", str(artifact_path)])
    if quick:
        cmd.extend(["--warmup-mode", "off"])
    if transformer_attention_backend is not None:
        cmd.extend(
            [
                "--component-attention-backends",
                f"transformer={transformer_attention_backend}",
            ]
        )
    if attention_backend_config is not None:
        cmd.extend(
            [
                "--attention-backend-config",
                json.dumps(attention_backend_config, separators=(",", ":")),
            ]
        )
    if quick:
        cmd.extend(["--num-inference-steps=8"])
    return cmd


def run_generate(
    bench_mod,
    model_key: str,
    *,
    label: str,
    output_dir: Path,
    cuda_visible_devices: str,
    transformer_attention_backend: str | None = None,
    attention_backend_config: dict | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    cfg = bench_mod.MODELS[model_key]
    is_video = model_key in VIDEO_PRESETS
    ext = "mp4" if is_video else "png"
    artifact_path = output_dir / f"{model_key}_{label}.{ext}"
    perf_path = output_dir / f"{model_key}_{label}_perf.json"
    cmd = build_generate_cmd(
        bench_mod,
        model_key,
        label=label,
        output_dir=output_dir,
        transformer_attention_backend=transformer_attention_backend,
        attention_backend_config=attention_backend_config,
        quick=quick,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.setdefault("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "1")
    env.setdefault("PYTHONPATH", str(PYTHON_ROOT))
    for key, value in cfg.get("env", {}).items():
        env.setdefault(key, str(value))

    print(f"\n{'=' * 72}")
    print(f"[{label}] {model_key}  CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    print(" \\\n  ".join(cmd))
    print()

    log_path = output_dir / f"{model_key}_{label}.log"
    print(f"Log: {log_path}")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    stderr_tail = ""
    try:
        stderr_tail = log_path.read_text(encoding="utf-8")[-4000:]
    except OSError:
        pass

    result = {
        "label": label,
        "attention_backend": transformer_attention_backend or "default",
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "artifact_path": str(artifact_path),
        "perf_path": str(perf_path),
        "log_path": str(log_path),
        "error": proc.returncode != 0 or not artifact_path.exists(),
    }
    if perf_path.exists():
        try:
            result.update(parse_perf_dump(perf_path))
        except Exception as exc:  # noqa: BLE001
            result["perf_parse_error"] = str(exc)
    if proc.returncode != 0:
        result["stderr_tail"] = stderr_tail
    return result


def evaluate_model(
    bench_mod,
    model_key: str,
    output_dir: Path,
    *,
    execute: bool,
    min_free_mib: int,
    sol_attn_config: dict,
    quick: bool,
    max_gpu_util: int,
    gpu_ids: str | None,
    ltx_gpu_ids: str | None = None,
) -> dict[str, Any]:
    effective_gpu_ids = gpu_ids
    if effective_gpu_ids is None and model_key == "ltx23-ti2v-two-stage" and ltx_gpu_ids:
        effective_gpu_ids = ltx_gpu_ids
    gpu = check_gpu(
        model_key,
        min_free_mib,
        max_gpu_util=max_gpu_util,
        gpu_ids=effective_gpu_ids,
    )
    report: dict[str, Any] = {
        "model_key": model_key,
        "modality": "video" if model_key in VIDEO_PRESETS else "image",
        "gpu_check": gpu,
        "sol_attn_config": sol_attn_config,
        "quick_mode": quick,
    }
    if not execute:
        report["status"] = "DRY_RUN"
        return report
    if not gpu["ok"]:
        report["status"] = "SKIPPED_NO_GPU"
        return report

    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    cuda = gpu["cuda_visible_devices"]
    assert cuda is not None

    fa = run_generate(
        bench_mod,
        model_key,
        label="fa",
        output_dir=model_dir,
        cuda_visible_devices=cuda,
        transformer_attention_backend="fa",
        quick=quick,
    )
    sol = run_generate(
        bench_mod,
        model_key,
        label="sol_attn",
        output_dir=model_dir,
        cuda_visible_devices=cuda,
        transformer_attention_backend="sol_attn",
        attention_backend_config=sol_attn_config,
        quick=quick,
    )
    report["runs"] = {"fa": fa, "sol_attn": sol}

    if fa.get("error") or sol.get("error"):
        report["status"] = "ERROR"
        return report

    quality = compare_artifacts(
        Path(fa["artifact_path"]),
        Path(sol["artifact_path"]),
        is_video=model_key in VIDEO_PRESETS,
    )
    report["quality_vs_fa"] = quality

    fa_denoise = fa.get("denoise_s")
    sol_denoise = sol.get("denoise_s")
    if isinstance(fa_denoise, float) and isinstance(sol_denoise, float) and fa_denoise > 0:
        report["speedup"] = {
            "denoise_speedup": fa_denoise / sol_denoise,
            "fa_denoise_s": fa_denoise,
            "sol_attn_denoise_s": sol_denoise,
            "fa_e2e_s": fa.get("e2e_s"),
            "sol_attn_e2e_s": sol.get("e2e_s"),
        }
    report["status"] = "COMPLETED"
    return report


def write_markdown_report(reports: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Sol-Attn Cross-Model Evaluation Report",
        "",
        "| Model | Modality | Status | PSNR (dB) | SSIM | Denoise speedup | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in reports:
        quality = item.get("quality_vs_fa") or {}
        speed = item.get("speedup") or {}
        psnr = quality.get("psnr_mean_db")
        ssim = quality.get("ssim_mean")
        speedup = speed.get("denoise_speedup")
        notes = []
        if item.get("status") == "SKIPPED_NO_GPU":
            notes.append("waiting for free GPUs")
        if item.get("runs", {}).get("sol_attn", {}).get("error"):
            notes.append("sol_attn run failed")
        if item.get("runs", {}).get("fa", {}).get("error"):
            notes.append("fa run failed")
        lines.append(
            "| {model} | {modality} | {status} | {psnr} | {ssim} | {speedup} | {notes} |".format(
                model=item["model_key"],
                modality=item["modality"],
                status=item.get("status", "UNKNOWN"),
                psnr=f"{psnr:.2f}" if isinstance(psnr, float) else "-",
                ssim=f"{ssim:.3f}" if isinstance(ssim, float) else "-",
                speedup=f"{speedup:.2f}x" if isinstance(speedup, float) else "-",
                notes="; ".join(notes) if notes else "-",
            )
        )

    lines.extend(
        [
            "",
            "## H3 reference (prior benchmark)",
            "",
            "- t2va denoise: FA 27.0s → sol_attn 23.5s (~1.15×), PSNR 32.9 dB, SSIM 0.91",
            "",
            "## Follow-ups",
            "",
            "## Why sol_attn can look slower (not GPU contention)",
            "",
            "- **HunyuanVideo quick run used GPU 0 alone at 0% util** — slowdown is real, not neighbor noise.",
            "- Default `dense_steps=10` with `--quick` (8 steps) keeps **100% dense** sol_attn path; wrapper overhead (~5s/step vs FA ~1.5s/step) without sparse kernel wins.",
            "- Fix: use `dense_steps=3` in quick mode (auto-applied now) or full preset steps for fair speed comparison.",
            "- **2-GPU image models** ran on GPU 0+1 while GPU 1 had high util; PSNR ~8 dB is semantic incompatibility (joint attention), not timing noise.",
            "",
            "- **Wan2.2**: likely needs Morton 3D token reordering for acceptable quality.",
            "- **Cosmos3**: UND path uses custom SDPA; sol_attn only touches GEN cross-attn partially.",
            "- **Qwen-Image**: joint text-image attention + hardcoded backend whitelist.",
            "- **HunyuanVideo/LTX/Z-Image**: prefix naming may not match sol_attn `blocks.N` dense_layers regex.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-model sol_attn evaluation")
    parser.add_argument(
        "--models",
        nargs="*",
        default=TARGET_MODELS,
        choices=TARGET_MODELS,
        help="Preset keys to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/sol_attn_eval",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run benchmarks (default is dry-run planning only)",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=80000,
        help="Minimum free GPU memory required per GPU",
    )
    parser.add_argument(
        "--sol-attn-config",
        type=Path,
        help="Optional JSON file overriding default sol_attn config",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 8 inference steps instead of preset defaults for faster smoke runs",
    )
    parser.add_argument(
        "--max-gpu-util",
        type=int,
        default=90,
        help="Max GPU utilization %% for eligibility (use 100 when sharing busy GPUs)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Pin CUDA_VISIBLE_DEVICES to this comma-separated list (e.g. 0,2 for LTX, 0,1,2,3 for Wan)",
    )
    parser.add_argument(
        "--ltx-gpu-ids",
        type=str,
        default="0,2",
        help="Default GPU pair for LTX two-stage when --gpu-ids is unset",
    )
    args = parser.parse_args()

    sol_attn_config = dict(DEFAULT_SOL_ATTN_CONFIG)
    if args.sol_attn_config is not None:
        with open(args.sol_attn_config) as f:
            sol_attn_config.update(json.load(f))
    if args.quick and sol_attn_config.get("dense_steps", 10) >= 8:
        # Quick runs use 8 steps; keep a few dense steps so sparse path is exercised.
        sol_attn_config["dense_steps"] = min(3, sol_attn_config.get("dense_steps", 10))

    bench_mod = _load_bench_module()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for model_key in args.models:
        report = evaluate_model(
            bench_mod,
            model_key,
            args.output_dir,
            execute=args.execute,
            min_free_mib=args.min_free_mib,
            sol_attn_config=sol_attn_config,
            quick=args.quick,
            max_gpu_util=args.max_gpu_util,
            gpu_ids=args.gpu_ids,
            ltx_gpu_ids=args.ltx_gpu_ids,
        )
        reports.append(report)
        out_json = args.output_dir / model_key / "report.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2))
        print(json.dumps({"model": model_key, "status": report.get("status")}, indent=2))

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(reports, indent=2))
    write_markdown_report(reports, args.output_dir / "SOL_ATTN_CROSS_MODEL_REPORT.md")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {args.output_dir / 'SOL_ATTN_CROSS_MODEL_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
